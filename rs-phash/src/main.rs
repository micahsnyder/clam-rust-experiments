extern crate image;

use anyhow::{anyhow, Result};
use image::imageops::FilterType::Lanczos3;
use rustdct::DctPlanner;
use std::path::{Path, PathBuf};
use structopt::StructOpt;
use transpose::transpose;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long)]
    debug: bool,

    #[structopt(name = "IMG_FILE", required = true)]
    img_path: Vec<PathBuf>,
}

pub fn main() -> Result<()> {
    let opt = Opt::from_args();
    let include_filename = opt.img_path.len() > 1;
    let debug = opt.debug;
    for path in &opt.img_path {
        print_phash_from_img_path(path, include_filename, debug)?;
    }
    Ok(())
}

/// Calculate a fuzzy image hash.
///
/// This algorithm attempts to reproduce the results of the `phash()` function
/// from the Python `imagehash` package.
///
/// # Notes
///
/// 1) I found that `image.grayscale() uses different RGB coefficients than
/// the python `image.convert("L"). The docs for PIL.Image.convert() state:
///
///     When translating a color image to greyscale (mode "L"),
///     the library uses the ITU-R 601-2 luma transform::
///
///         L = R * 299/1000 + G * 587/1000 + B * 114/1000
///
/// You can get near-identical** grayscale results by making a clone (or forking)
/// the image-rs crate, and changing the coefficients to match those above:
///
///     diff --git a/src/color.rs b/src/color.rs
///     index 78b5c587..92c99337 100644
///     --- a/src/color.rs
///     +++ b/src/color.rs
///     @@ -462,7 +462,7 @@ where
///      }
///
///      /// Coefficients to transform from sRGB to a CIE Y (luminance) value.
///     -const SRGB_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
///     +const SRGB_LUMA: [f32; 3] = [0.299, 0.587, 0.114];
///
///      #[inline]
///      fn rgb_to_luma<T: Primitive>(rgb: &[T]) -> T {
///
/// This change isn't really required, but it helps when debugging to determine
/// differences between the implementations.
///
/// **Note that I say "near-identical" because rounding
/// appears to be slightly different and values are sometimes off-by-one.
///
/// 2) scipy.fftpack.dct behaves differently on twodimensional arrays than
/// single-dimensional arrays.
/// See https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html:
///
///     Note the optional "axis" argument:
///         Axis along which the dct is computed; the default is over the last axis
///         (i.e., axis=-1).
///
/// For the Python `imagehash` package:
/// - The `phash_simple()` function is doing a DCT-2 transform on a 2-dimensionals
/// 32x32 array which means, just on the 2nd axis (just the rows).
/// - The `phash()` function is doing a 2D DCT-2 transform, by running the DCT-2 on
/// both X and Y axis, which is the same as transposing before or after each
/// DCT-2 call.
///
/// 3) I observed that the DCT2 results from Python are consistently 2x greater
/// than those from Rust. If I multiply every value by 2 after running the DCT,
/// then the results are the same.
///
/// 4) We need to get a subset of the 2-D array representing the lower
/// frequencies of the image, the same way the Python implementation does it.
///
/// The way the python implementation does this is with this line:
/// ```python
/// dctlowfreq = dct[:hash_size, :hash_size]
/// ```
///
/// You can't actually do that with a Python array of arrays... this is numpy
/// 2-D array manipulation magic, where you can index 2-D arrays in slices.
/// It works like this:
/// ```ipython3
/// In [0]: x = [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8], [8, 9, 10, 11, 12], [12, 13, 14, 15, 16], [16, 17, 18, 19, 20]]
/// In [1]: h = 3
/// In [2]: n = np.asarray(x)
/// In [3]: lf = n[:h, 1:h+1]
/// In [4]: n
/// array([[ 0,  1,  2,  3,  4],
///        [ 4,  5,  6,  7,  8],
///        [ 8,  9, 10, 11, 12],
///        [12, 13, 14, 15, 16],
///        [16, 17, 18, 19, 20]])
///
/// In [5]: lf
/// array([[ 0,  1,  2],
///        [ 4,  5,  6],
///        [ 8,  9, 10]])
/// ```
///
/// We can do something similar, manually, to get the low-frequency selection.
fn print_phash_from_img_path(path: &Path, include_filename: bool, debug: bool) -> Result<()> {
    // Open image given a file path.
    let image = image::io::Reader::open(&path)
        .map_err(|e| anyhow!("opening {:?}: {}", &path, e))?
        // Guess at the format, in case the file extension lies.
        .with_guessed_format()
        .map_err(|e| anyhow!("guessing type {:?}: {}", &path, e))?
        // Parse the image.
        .decode()
        .map_err(|e| anyhow!("decoding {:?}: {}", &path, e))?;

    // Convert image to grayscale.
    let image_gs = image.grayscale();

    // Shrink to a 32x32 (1024 pixel) image.
    let image_small = image::DynamicImage::resize_exact(&image_gs, 32, 32, Lanczos3);

    // Drop the alpha channel.
    let image_rgb8 = image_small.to_luma8();

    // Convert the data to a Vec of 64-bit floats.
    let imgbuff_u8 = image_rgb8.to_vec();
    let mut imgbuff_f64: Vec<f64> = imgbuff_u8.into_iter().map(|x| x as f64).collect();

    //
    // Compute a 2D DCT-2 in-place.
    //
    let dct2 = DctPlanner::new().plan_dct2(32);

    // Use a scratch space so we can transpose and run DCT's without allocating any extra space.
    // We'll switch back and forth between the buffer for the original small image (buffer1) and the scratch buffer (buffer2).
    let mut buffer1: &mut [f64] = imgbuff_f64.as_mut_slice();
    let mut buffer2: &mut [f64] = &mut [0.0; 1024];

    // Transpose the image so we can run DCT on the X axis (columns) first.
    transpose(buffer1, &mut buffer2, 32, 32);

    // Run DCT2 on the columns.
    for (row_in, row_out) in buffer2.chunks_mut(32).zip(buffer1.chunks_mut(32)) {
        dct2.process_dct2_with_scratch(row_in, row_out);
    }
    // Multiply each value x2, to match results from scipy.fftpack.dct() implementation.
    // Note: Unsure why this is required, but it is.
    buffer2.iter_mut().for_each(|f| *f *= 2.0);

    // Transpose the image back so we can run DCT on the Y axis (rows).
    transpose(buffer2, &mut buffer1, 32, 32);

    // Run DCT2 on the rows.
    for (row_in, row_out) in buffer1.chunks_mut(32).zip(buffer2.chunks_mut(32)) {
        dct2.process_dct2_with_scratch(row_in, row_out);
    }
    // Multiply each value x2, to match results from scipy.fftpack.dct() implementation.
    // Note: Unsure why this is required, but it is.
    buffer1.iter_mut().for_each(|f| *f *= 2.0);

    //
    // Construct a DCT low frequency vector using the top-left most 8x8 values of the 32x32 DCT array.
    //
    let dct_low_freq = buffer1
        // 2D array is 32-elements wide.
        .chunks(32)
        // Grab the first 8 rows.
        .take(8)
        // But only take the first 8 elements (columns) from each row.
        .map(|chunk| chunk.chunks(8).take(1))
        // Flatten the 8x8 selection down to a vector of f64's.
        .flatten()
        .flatten()
        .copied()
        .collect::<Vec<f64>>();

    // Calculate average (median) of the DCT low frequency vector.
    let mut dct_low_freq_copy = dct_low_freq.clone();
    dct_low_freq_copy.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = dct_low_freq_copy.len() / 2;
    let median = dct_low_freq_copy[mid];

    // Construct hash vector by reducing DCT values to 1 or 0 by comparing terms vs median.
    let hashvec: Vec<u64> = dct_low_freq
        .into_iter()
        .map(|x| if x > median { 1 } else { 0 })
        .collect();

    // Construct hash integer from bits.
    let hash: u64 = hashvec.iter().fold(0, |res, &bit| (res << 1) ^ bit);

    if include_filename && debug {
        println!("{:?}: {:08x}", &path, hash);
    } else {
        println!("{:08x}", hash);
    }
    Ok(())
}
