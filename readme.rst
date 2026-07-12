Description
===========

znedi3 is a CPU-optimized version of nnedi.

nnedi3 is an intra-field only deinterlacer. It takes a frame, throws
away one field, and then interpolates the missing pixels using only
information from the remaining field. It is also good for enlarging
images by powers of two.

This plugin no longer provides the nnedi3_rpow2 filter. A replacement
can be found here: http://forum.doom9.org/showthread.php?t=172652

This is a port of tritical's nnedi3 filter.


Usage
=====

The file ``nnedi3_weights.bin`` is always required and must be in the
same directory as nnedi3. ``nnedi3_weights_hvsym.bin`` must be there too
unless *new_weights* is False. Both are loaded automatically; a missing
required file is an error.

When *new_weights* is True (the default), the two largest *nns* settings
(128 and 256 neurons) use a folded predictor that exploits the network's
horizontal and vertical mirror symmetry to run roughly 4x less
arithmetic. It is a symmetrised approximation: most output pixels are
identical to the original predictor and the mean difference is well
under one 8-bit code value, but individual pixels can differ by more
(tens of code values in the worst case). Configurations the folded file
does not cover fall back to the original predictor automatically. Set
*new_weights* to False to use the original predictor everywhere.

::

   nnedi3.nnedi3(clip clip, int field[, bint dh=False, int[] planes=[0, 1, 2], int nsize=6, int nns=1, int qual=1, int etype=0, int pscrn=2, bint opt=True, bint int16_prescreener=True, bint int16_predictor=True, int exp=0, bint show_mask=False, bint new_weights=True, bint silent_fallback=True])

Parameters:
    *clip*
        Clip to process. It must have constant format and dimensions,
        and integer samples with 8..16 bits or float samples with 32
        bits.

    *field*
        Selects the mode of operation. Possible values:

        * 0: Same rate, keep bottom field.
        * 1: Same rate, keep top field.
        * 2: Double rate, start with bottom field.
        * 3: Double rate, start with top field.

        If *dh* is True, the ``_Field`` frame property is used to
        determine each frame's field dominance. The *field* parameter
        is only a fallback for frames that don't have the ``_Field``
        property.

        If *dh* is False, the ``_FieldBased`` frame property is used
        to determine each frame's field dominance. The *field*
        parameter is only a fallback for frames that don't have the
        ``_FieldBased`` property, or where said property indicates
        that the frame is progressive.

    *dh*
        Doubles the height, keeping both fields. If *field* is 0, the
        input is copied to the odd lines of the output (the bottom
        field). If *field* is 1, the input is copied to the even lines
        of the output (the top field).

        If *dh* is True, *field* must be 0 or 1.

        Default: False.

    *planes*
        Planes to process. Planes that are not processed will contain
        uninitialised memory.

        Default: all.

    *nsize*
        Size of the local neighbourhood around each pixel used by the
        predictor neural network. Possible settings:

        * 0: 8x6
        * 1: 16x6
        * 2: 32x6
        * 3: 48x6
        * 4: 8x4
        * 5: 16x4
        * 6: 32x4

        For image enlargement it is recommended to use 0 or 4. A taller
        neighbourhood will result in sharper output.

        For deinterlacing a wider neighbourhood will allow connecting
        lines of smaller slope. However, the setting to use depends on
        the amount of aliasing (lost information) in the source. If
        the source was heavily low-pass filtered before interlacing
        then aliasing will be low and a wide neighbourhood won't be
        needed, and vice-versa.

        Default: 6.

    *nns*
        Number of neurons in the predictor neural network. Possible
        values:

        * 0: 16
        * 1: 32
        * 2: 64
        * 3: 128
        * 4: 256

        Higher values are slower, but provide better quality. However,
        quality differences are usually small. The difference in speed
        will become larger if *qual* is increased.

        Default: 1.

    *qual*
        The number of different neural network predictions that are
        blended together to compute the final output value. Each
        neural network was trained on a different set of training
        data. Blending the results of these different networks
        improves generalisation to unseen data. Possible values are
        1 and 2.

        A value of 2 is recommended for image enlargement.

        Default: 1.

    *etype*
        The set of weights used in the predictor neural network.
        Possible values:

        * 0: Weights trained to minimise absolute error.
        * 1: Weights trained to minimise squared error.

        Default: 0.

    *pscrn*
        The prescreener used to decide which pixels should be
        processed by the predictor neural network, and which can be
        handled by simple cubic interpolation. Since most pixels can
        be handled by cubic interpolation, using the prescreener
        generally results in much faster processing. Possible values:

        * 0: No prescreening. No pixels will be processed with cubic
          interpolation. This is really slow.
        * 1: Old prescreener.
        * 2: New prescreener level 0.
        * 3: New prescreener level 1.
        * 4: New prescreener level 2.

        The new prescreener works faster than the old one, and it also
        causes more pixels to be processed with cubic interpolation.
        The higher levels cause a bit more pixels to be processed with
        the predictor neural network, therefore they are slower than
        the lowest level.

        The new prescreener is not available with float input.

        Default: 2 for integer input, 1 for float input.

    *opt*
        If True, the best optimised functions supported by the CPU
        will be used. If False, only scalar functions will be used.

        Default: True.

    *int16_prescreener*
        If True, the prescreener will perform the dot product
        calculations using 16 bit integers. Otherwise, it will use
        single precision floats.

        This parameter is ignored when the input has float samples.

        Default: True.

    *int16_predictor*
        If True, the predictor performs its matrix-vector products
        using 16 bit integers (an int16 VNNI kernel, roughly twice the
        throughput of the float path). It applies both to the plain
        predictor and to the automatic folded predictor.

        It requires automatic CPU dispatch (*opt* True and no *x_cpu*
        override) on a CPU with AVX-512 and AVX-512-VNNI. On any other
        CPU the float path is used transparently.

        Unlike the float path it is a small precision reduction. On
        integer input (8..16 bits) the reduction is benign (well below
        the network's own prediction error). On float or half input it
        would be a genuine downgrade, so with *silent_fallback* False it
        raises an error there; with *silent_fallback* True (the default)
        it is simply ignored.

        Default: True.

    *exp*
        The exp function approximation to use in the predictor. 0 is
        the fastest and least accurate. 2 is the slowest and most
        accurate.

        Default: 0.

    *show_mask*
        If True, the pixels that would be processed with the predictor
        neural network are instead set to white.

        Default: False.

    *new_weights*
        Selects between the two weight sets. When True (the default) the
        folded h+v-symmetric predictor from ``nnedi3_weights_hvsym.bin``
        is used for the configurations it covers (the two largest *nns*
        settings), which is faster but a symmetrised approximation of the
        original network (see above). When False the original dense
        predictor from ``nnedi3_weights.bin`` is used everywhere, and
        ``nnedi3_weights_hvsym.bin`` is not needed.

        Default: True.
        Governs what happens when an acceleration cannot be used for a
        reason other than CPU capability or a missing weights file. When
        True (the default) the filter silently uses the slower path
        instead:

        * a configuration the folded file does not cover, or that is
          below the *nns* gate, uses the plain predictor;
        * *int16_predictor* on float or half input is ignored.

        When False such a case is a hard error instead, so you can tell
        that the fast path did not engage.

        Unavailability caused by the CPU lacking AVX-512-VNNI always
        falls back to the float path silently, regardless of this
        setting. A missing weights file is always an error, regardless
        of this setting.

        Default: True.


Compilation (Linux)
===================

Clone the repository (using the :code:`--recursive` argument to also dowload the required :code:`vsxx` library as a submodule):
::

   git clone --recursive https://github.com/sekrit-twc/znedi3

Compile the library:
::

   cd znedi3
   make X86=1

   
To install, copy :code:`vsznedi3.so`, :code:`nnedi3_weights.bin` and (optionally, for the folded predictor) :code:`nnedi3_weights_hvsym.bin` to the vapoursynth plugin folder (usually :code:`/usr/lib/x86_64-linux-gnu/vapoursynth/`):
::

   sudo cp nnedi3_weights.bin nnedi3_weights_hvsym.bin vsznedi3.so /usr/lib/x86_64-linux-gnu/vapoursynth/

There is also a test application which can be built to check the efficiency of the plugin kernels optimized for different SIMD instructions:
::

   make X86=1 testapp/testapp

License
=======

GPLv2.
