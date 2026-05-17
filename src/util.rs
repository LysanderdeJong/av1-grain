use std::slice::SliceIndex;
#[cfg(feature = "diff")]
use std::{borrow::Cow, mem::size_of};

#[cfg(feature = "diff")]
use rayon::{prelude::*, scope};
#[cfg(feature = "diff")]
use v_frame::{frame::Frame, pixel::Pixel, plane::Plane};

use cfg_if::cfg_if;

#[cfg(feature = "diff")]
pub fn frame_into_u8<T: Pixel + Send + Sync>(
    frame: &Frame<T>,
    bit_depth: usize,
) -> Cow<'_, Frame<u8>> {
    if size_of::<T>() == 1 {
        assert_eq!(bit_depth, 8);
        // SAFETY: We know from the size check that this must be a `Frame<u8>`
        Cow::Borrowed(unsafe { &*(frame as *const Frame<T>).cast::<Frame<u8>>() })
    } else if size_of::<T>() == 2 {
        use std::num::NonZeroU8;

        use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder};

        assert!(bit_depth > 8 && bit_depth <= 16);
        let mut u8_frame: Frame<u8> = FrameBuilder::new(
            frame.y_plane.width(),
            frame.y_plane.height(),
            frame.subsampling,
            NonZeroU8::new(8).expect("non-zero constant"),
        )
        .build()
        .expect("frame should build");
        let shift = bit_depth - 8usize;
        if frame.subsampling == ChromaSubsampling::Monochrome {
            convert_plane_to_u8(&frame.y_plane, &mut u8_frame.y_plane, shift);
        } else {
            let in_u = frame
                .u_plane
                .as_ref()
                .expect("unreachable due to subsampling check");
            let in_v = frame
                .v_plane
                .as_ref()
                .expect("unreachable due to subsampling check");
            let out_u = u8_frame
                .u_plane
                .as_mut()
                .expect("unreachable due to subsampling check");
            let out_v = u8_frame
                .v_plane
                .as_mut()
                .expect("unreachable due to subsampling check");

            scope(|s| {
                s.spawn(|_| convert_plane_to_u8(&frame.y_plane, &mut u8_frame.y_plane, shift));
                s.spawn(|_| convert_plane_to_u8(in_u, out_u, shift));
                s.spawn(|_| convert_plane_to_u8(in_v, out_v, shift));
            });
        }
        Cow::Owned(u8_frame)
    } else {
        unimplemented!("Bit depths greater than 16 are not currently supported");
    }
}

#[cfg(feature = "diff")]
fn convert_plane_to_u8<T: Pixel>(in_plane: &Plane<T>, out_plane: &mut Plane<u8>, shift: usize) {
    debug_assert_eq!(size_of::<T>(), 2);

    let in_geometry = in_plane.geometry();
    let out_geometry = out_plane.geometry();
    debug_assert_eq!(in_geometry.width, out_geometry.width);
    debug_assert_eq!(in_geometry.height, out_geometry.height);

    let width = in_geometry.width.get();
    let height = in_geometry.height.get();
    let in_stride = in_geometry.stride.get();
    let out_stride = out_geometry.stride.get();
    let in_row_start = in_geometry.stride.get() * in_geometry.pad_top;
    let out_row_start = out_geometry.stride.get() * out_geometry.pad_top;

    // SAFETY: Pixel is implemented only for u8/u16; this function is called only
    // from the high-bit-depth branch, so T is u16.
    let in_data = unsafe {
        std::slice::from_raw_parts(
            in_plane.data().as_ptr().cast::<u16>(),
            in_plane.data().len(),
        )
    };
    let in_rows = get_dbg(in_data, in_row_start..);
    let out_rows = get_dbg_mut(out_plane.data_mut(), out_row_start..);

    out_rows
        .par_chunks_mut(out_stride)
        .take(height)
        .enumerate()
        .for_each(|(y, out_row)| {
            let in_visible_start = y * in_stride + in_geometry.pad_left;
            let in_row = get_dbg(in_rows, in_visible_start..in_visible_start + width);
            let out_start = out_geometry.pad_left;
            let out_row = get_dbg_mut(out_row, out_start..out_start + width);
            for (i, o) in in_row.iter().zip(out_row.iter_mut()) {
                *o = (*i >> shift) as u8;
            }
        });
}

#[allow(
    clippy::inline_always,
    reason = "intended as a thin compile-time-elided wrapper"
)]
#[inline(always)]
pub fn get_dbg<T, I: SliceIndex<[T]>>(arr: &[T], index: I) -> &<I as SliceIndex<[T]>>::Output {
    cfg_if! {
        if #[cfg(debug_assertions)] {
            arr.get(index).expect("array index out of bounds")
        } else {
            unsafe{ arr.get_unchecked(index) }
        }
    }
}

#[allow(
    clippy::inline_always,
    reason = "intended as a thin compile-time-elided wrapper"
)]
#[inline(always)]
pub fn get_dbg_mut<T, I: SliceIndex<[T]>>(
    arr: &mut [T],
    index: I,
) -> &mut <I as SliceIndex<[T]>>::Output {
    cfg_if! {
        if #[cfg(debug_assertions)] {
            arr.get_mut(index).expect("array index out of bounds")
        } else {
            unsafe{ arr.get_unchecked_mut(index) }
        }
    }
}

#[cfg(all(test, feature = "diff"))]
mod tests {
    use std::num::{NonZeroU8, NonZeroUsize};

    use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder};

    use super::frame_into_u8;

    #[test]
    fn frame_into_u8_preserves_visible_high_bit_depth_pixels_with_padding() {
        let mut frame = FrameBuilder::new(
            NonZeroUsize::new(8).expect("non-zero constant"),
            NonZeroUsize::new(4).expect("non-zero constant"),
            ChromaSubsampling::Yuv420,
            NonZeroU8::new(10).expect("non-zero constant"),
        )
        .luma_padding_left(2)
        .luma_padding_right(4)
        .luma_padding_top(2)
        .luma_padding_bottom(2)
        .build::<u16>()
        .expect("valid frame");

        for plane_index in 0..3 {
            let plane = frame.plane_mut(plane_index).expect("plane exists");
            for (index, pixel) in plane.pixels_mut().enumerate() {
                *pixel = ((index * 37 + plane_index * 101) % 1024) as u16;
            }
        }

        let converted = frame_into_u8(&frame, 10);
        for plane_index in 0..3 {
            let input = frame.plane(plane_index).expect("plane exists");
            let output = converted.plane(plane_index).expect("plane exists");
            for (i, o) in input.pixels().zip(output.pixels()) {
                assert_eq!((i >> 2) as u8, o);
            }
        }
    }
}
