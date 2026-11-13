"""
PIL formats for multiple images.
"""

import logging

import numpy as np

from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil

logger = logging.getLogger(__name__)

NeuQuant = None  # we can implement this when we need it


class TIFFFormat(PillowFormat):
    _modes = "i"  # arg, why bother; people should use the tiffile version
    _description = "TIFF format (Pillow)"


class GIFFormat(PillowFormat):
    """See :mod:`imageio.plugins.pillow_legacy`"""

    _modes = "iI"
    _description = "Static and animated gif (Pillow)"

    # GIF reader needs no modifications compared to base pillow reader

    class Writer(PillowFormat.Writer):
        def _open(
            self,
            loop=0,
            duration=None,
            fps=10,
            palettesize=256,
            quantizer=0,
            subrectangles=False,
        ):
            # Check palettesize
            palettesize = int(palettesize)
            if palettesize < 2 or palettesize > 256:
                raise ValueError("GIF quantize param must be 2..256")
            if palettesize not in [2, 4, 8, 16, 32, 64, 128, 256]:
                palettesize = 2 ** int(np.log2(128) + 0.999)
                logger.warning(
                    "Warning: palettesize (%r) modified to a factor of "
                    "two between 2-256." % palettesize
                )
            #  Duratrion / fps
            if duration is None:
                self._duration = 1.0 / float(fps)
            elif isinstance(duration, (list, tuple)):
                self._duration = [float(d) for d in duration]
            else:
                self._duration = float(duration)
            # loop
            loop = float(loop)
            if loop <= 0 or loop == float("inf"):
                loop = 0
            loop = int(loop)
            # Subrectangles / dispose
            subrectangles = bool(subrectangles)
            self._dispose = 1 if subrectangles else 2
            # The "0" (median cut) quantizer is by far the best

            fp = self.request.get_file()
            self._writer = GifWriter(
                fp, subrectangles, loop, quantizer, int(palettesize)
            )

        def _close(self):
            self._writer.close()

        def _append_data(self, im, meta):
            im = image_as_uint(im, bitdepth=8)
            if im.ndim == 3 and im.shape[-1] == 1:
                im = im[:, :, 0]
            duration = self._duration
            if isinstance(duration, list):
                duration = duration[min(len(duration) - 1, self._writer._count)]
            dispose = self._dispose
            self._writer.add_image(im, duration, dispose)

            return


def intToBin(i):
    return i.to_bytes(2, byteorder="little")


def _normalize_color_table(palette):
    """Determine the GIF color table size for ``palette``.

    A GIF color table must contain a power-of-two number of entries (2 to
    256) and the packed field in the header/descriptor stores
    ``log2(entries) - 1`` in its lower three bits. Pillow usually returns a
    padded palette, but we derive the size from the palette we actually
    write and pad/truncate it to match. Otherwise the header advertises a
    table that is not there, and decoders read the image data as palette
    bytes, which corrupts the file.

    Returns ``(palette, table_size)`` where ``table_size`` is the value to
    OR into the packed field.
    """
    palette = bytes(palette)
    # palettes are sequences of RGB triples; drop any trailing partial entry
    palette = palette[: (len(palette) // 3) * 3]
    n_colors = max(len(palette) // 3, 1)
    # round up to the next power of two, with a minimum of two and a
    # maximum of 256 entries (the largest table a GIF supports)
    n_entries = min(1 << max((n_colors - 1).bit_length(), 1), 256)

    target_length = n_entries * 3
    if len(palette) < target_length:
        palette += b"\x00" * (target_length - len(palette))
    elif len(palette) > target_length:
        palette = palette[:target_length]

    return palette, n_entries.bit_length() - 2


class GifWriter:
    """Class that for helping write the animated GIF file. This is based on
    code from images2gif.py (part of visvis). The version here is modified
    to allow streamed writing.
    """

    def __init__(
        self,
        file,
        opt_subrectangle=True,
        opt_loop=0,
        opt_quantizer=0,
        opt_palette_size=256,
    ):
        self.fp = file

        self.opt_subrectangle = opt_subrectangle
        self.opt_loop = opt_loop
        self.opt_quantizer = opt_quantizer
        self.opt_palette_size = opt_palette_size

        self._previous_image = None  # as np array
        self._global_palette = None  # as bytes
        self._count = 0

        from PIL.GifImagePlugin import getdata

        self.getdata = getdata

    def add_image(self, im, duration, dispose):
        # Prepare image
        im_rect, rect = im, (0, 0)
        if self.opt_subrectangle:
            im_rect, rect = self.getSubRectangle(im)
        im_pil = self.converToPIL(im_rect, self.opt_quantizer, self.opt_palette_size)

        # Get the palette Pillow would write. Depending on the Pillow
        # version the palette lives either in the header element (Pillow
        # pads it to a power of two) or in the image's palette object (raw,
        # and RGBA for an RGBA palette mode), so prefer the raw bytes and
        # fall back to the header.
        from PIL.GifImagePlugin import getheader

        header = getheader(im_pil)[0]
        palette = bytes(im_pil.palette.palette) if im_pil.palette else b""
        if im_pil.palette is not None and im_pil.palette.mode == "RGBA":
            palette = b"".join(
                palette[i : i + 3] for i in range(0, len(palette), 4)
            )
        if not palette:
            palette = header[3] if len(header) > 3 else b""
        # Make sure the color table we write matches the size we advertise
        # in the header and in the image descriptor. Otherwise decoders read
        # image data as palette bytes, which corrupts the file.
        palette, table_size = _normalize_color_table(palette)

        # Write image
        if self._count == 0:
            self.write_header(im_pil, palette, table_size, self.opt_loop)
            self._global_palette = palette
        self.write_image(im_pil, palette, table_size, rect, duration, dispose)
        # assert len(palette) == len(self._global_palette)

        # Bookkeeping
        self._previous_image = im
        self._count += 1

    def write_header(self, im, globalPalette, table_size, loop):
        # Gather info
        header = self.getheaderAnim(im, table_size)
        appext = self.getAppExt(loop)
        # Write
        self.fp.write(header)
        self.fp.write(globalPalette)
        self.fp.write(appext)

    def close(self):
        self.fp.write(";".encode("utf-8"))  # end gif

    def write_image(self, im, palette, table_size, rect, duration, dispose):
        fp = self.fp

        # Gather local image header and data, using PIL's getdata. That
        # function returns a list of bytes objects, but which parts are
        # what has changed multiple times, so we put together the first
        # parts until we have enough to form the image header.
        data = self.getdata(im)
        imdes = b""
        while data and len(imdes) < 11:
            imdes += data.pop(0)
        assert len(imdes) == 11

        # Make image descriptor suitable for using a local color palette
        lid = self.getImageDescriptor(im, table_size, rect)
        graphext = self.getGraphicsControlExt(duration, dispose)

        # Write local header
        if (palette != self._global_palette) or (dispose != 2):
            # Use local color palette
            fp.write(graphext)
            fp.write(lid)  # write suitable image descriptor
            fp.write(palette)  # write local color table
            fp.write(b"\x08")  # LZW minimum size code
        else:
            # Use global color palette
            fp.write(graphext)
            fp.write(imdes)  # write suitable image descriptor

        # Write image data
        for d in data:
            fp.write(d)

    def getheaderAnim(self, im, table_size=7):
        """Get animation header. To replace PILs getheader()[0]"""
        bb = b"GIF89a"
        bb += intToBin(im.size[0])
        bb += intToBin(im.size[1])
        # packed field: global color table flag1, color resolution, sorted
        # table0, size of global color table (log2(entries) - 1)
        bb += bytes([0x80 | table_size]) + b"\x00\x00"
        return bb

    def getImageDescriptor(self, im, table_size=7, xy=None):
        """Used for the local color table properties per image.
        Otherwise global color table applies to all frames irrespective of
        whether additional colors comes in play that require a redefined
        palette. Still a maximum of 256 color per frame, obviously.

        Written by Ant1 on 2010-08-22
        Modified by Alex Robinson in Janurari 2011 to implement subrectangles.
        """

        # Default: use full image and place at upper left
        if xy is None:
            xy = (0, 0)

        # Image separator,
        bb = b"\x2c"

        # Image position and size
        bb += intToBin(xy[0])  # Left position
        bb += intToBin(xy[1])  # Top position
        bb += intToBin(im.size[0])  # image width
        bb += intToBin(im.size[1])  # image height

        # packed field: local color table flag1, interlace0, sorted table0,
        # reserved00, size of local color table (log2(entries) - 1).
        bb += bytes([0x80 | table_size])

        # LZW minimum size code now comes later, beginning of [imagedata] blocks
        return bb

    def getAppExt(self, loop):
        """Application extension. This part specifies the amount of loops.
        If loop is 0 or inf, it goes on infinitely.
        """
        if loop == 1:
            return b""
        if loop == 0:
            loop = 2**16 - 1
        bb = b""
        if loop != 0:  # omit the extension if we would like a nonlooping gif
            bb = b"\x21\xff\x0b"  # application extension
            bb += b"NETSCAPE2.0"
            bb += b"\x03\x01"
            bb += intToBin(loop)
            bb += b"\x00"  # end
        return bb

    def getGraphicsControlExt(self, duration=0.1, dispose=2):
        """Graphics Control Extension. A sort of header at the start of
        each image. Specifies duration and transparency.

        Dispose
        -------
          * 0 - No disposal specified.
          * 1 - Do not dispose. The graphic is to be left in place.
          * 2 - Restore to background color. The area used by the graphic
            must be restored to the background color.
          * 3 - Restore to previous. The decoder is required to restore the
            area overwritten by the graphic with what was there prior to
            rendering the graphic.
          * 4-7 -To be defined.
        """

        bb = b"\x21\xf9\x04"
        bb += chr((dispose & 3) << 2).encode("utf-8")
        # low bit 1 == transparency,
        # 2nd bit 1 == user input , next 3 bits, the low two of which are used,
        # are dispose.
        bb += intToBin(int(duration * 100 + 0.5))  # in 100th of seconds
        bb += b"\x00"  # no transparent color
        bb += b"\x00"  # end
        return bb

    def getSubRectangle(self, im):
        """Calculate the minimal rectangle that need updating. Returns
        a two-element tuple containing the cropped image and an x-y tuple.

        Calculating the subrectangles takes extra time, obviously. However,
        if the image sizes were reduced, the actual writing of the GIF
        goes faster. In some cases applying this method produces a GIF faster.
        """

        # Cannot do subrectangle for first image
        if self._count == 0:
            return im, (0, 0)

        prev = self._previous_image

        # Get difference, sum over colors
        diff = np.abs(im - prev)
        if diff.ndim == 3:
            diff = diff.sum(2)
        # Get begin and end for both dimensions
        X = np.argwhere(diff.sum(0))
        Y = np.argwhere(diff.sum(1))
        # Get rect coordinates
        if X.size and Y.size:
            x0, x1 = int(X[0][0]), int(X[-1][0] + 1)
            y0, y1 = int(Y[0][0]), int(Y[-1][0] + 1)
        else:  # No change ... make it minimal
            x0, x1 = 0, 2
            y0, y1 = 0, 2

        return im[y0:y1, x0:x1], (x0, y0)

    def converToPIL(self, im, quantizer, palette_size=256):
        """Convert image to Paletted PIL image.

        PIL used to not do a very good job at quantization, but I guess
        this has improved a lot (at least in Pillow). I don't think we need
        neuqant (and we can add it later if we really want).
        """

        im_pil = ndarray_to_pil(im, "gif")

        if quantizer in ("nq", "neuquant"):
            # NeuQuant algorithm
            nq_samplefac = 10  # 10 seems good in general
            im_pil = im_pil.convert("RGBA")  # NQ assumes RGBA
            nqInstance = NeuQuant(im_pil, nq_samplefac)  # Learn colors
            im_pil = nqInstance.quantize(im_pil, colors=palette_size)
        elif quantizer in (0, 1, 2):
            # Adaptive PIL algorithm
            if quantizer == 2:
                im_pil = im_pil.convert("RGBA")
            else:
                im_pil = im_pil.convert("RGB")
            im_pil = im_pil.quantize(colors=palette_size, method=quantizer)
        else:
            raise ValueError("Invalid value for quantizer: %r" % quantizer)
        return im_pil
