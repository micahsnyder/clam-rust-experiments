extern crate image;

use anyhow::{anyhow, Result};
use image::imageops::FilterType::Lanczos3;
use rustdct::DctPlanner;
use std::path::{Path, PathBuf};
use structopt::StructOpt;

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

/// # Modifications made to Mickey's algorithm
///
/// 1️⃣First, I found that `image.grayscale() uses different RGB coefficients than
/// the python `image.convert("L"). The docs for PIL.Image.convert() state:
///
///     When translating a color image to greyscale (mode "L"),
///     the library uses the ITU-R 601-2 luma transform::
///
///         L = R * 299/1000 + G * 587/1000 + B * 114/1000
///
/// In a local clone of the image-rs crate, this change manages near-identical
/// results to the Python "L" mode:
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
/// I don't think this👆 change is really required⭐, but it helps for determining
/// how things change later on. Note that I say "near-identical" because rounding
/// appears to be slightly different and values are sometimes off-by-one.
///
/// ⭐I tested later using the original image crate without the greyscale()
/// modifications and it does indeed provide the same "phash" in the end. 👍
///
/// 2️⃣Second, I found that the Python imagehash.phash_simple() function is doing a
/// DCT-2 transform on the a 2-dimensionals 32x32 array. scipy.fftpack.dct behaves
/// differently on twodimensional arrays than single-dimensional arrays.
///
/// Read https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
///
/// Note the optional "axis" argument:
///     Axis along which the dct is computed; the default is over the last axis
///     (i.e., axis=-1).
///
/// With this change, I was able to get identical DCT results to the Rust DCT-2
/// of the flat 1024 byte array:
///
///     diff --git a/imagehash.py b/imagehash.py
///     index 31d6307..0ebe12c 100644
///     --- a/imagehash.py
///     +++ b/imagehash.py
///     @@ -228,7 +228,9 @@ def phash_simple(image, hash_size=8, highfreq_factor=4):
///             img_size = hash_size * highfreq_factor
///             image = image.convert("L").resize((img_size, img_size), Image.ANTIALIAS)
///             pixels = numpy.asarray(image)
///     +       pixels = pixels.flatten()
///             dct = scipy.fftpack.dct(pixels)
///     +       dct = list(map(lambda x: x/2, dct))
///             dctlowfreq = dct[:hash_size, 1:hash_size+1]
///             avg = dctlowfreq.mean()
///             diff = dctlowfreq > avg
///
/// I'm not sure why I had to divide each value by 2.
///
/// But of course we don't want to modify the Python one. I'm just showing that
/// the DCT-2 calculation works the same when flattened, but is different on the
/// 2-dimensional array.
///
/// What we need to do is compute the DCT-2 in Rust over the last axis, the same
/// way it is done in Python when we don't flatten the array. We can do this by
/// doing a DCT of each 32 byte chunk:
///
/// ```rust
/// let dct2 = DctPlanner::new().plan_dct2(32);
/// for mut chunk in imgbuff_f64.chunks_mut(32) {
///     dct2.process_dct2(&mut chunk);
/// }
///
/// let _: () = imgbuff_f64
///     .iter_mut()
///     .map(|f| {
///         *f = *f * 2.0;
///     })
///     .collect();
/// ```
/// As before, we have to multiply each value by 2.0 to match the DCT2 results
/// seen in the Python scipy.fftpack.dct() implementation. 🤷‍♀️
///
/// 3️⃣Third, we need to get a subset of the 2-D array representing the lower
/// frequencies of the image, the same way the Python implementation does it.
///
/// The way the python implementation does this is with this line:
/// ```python
/// dctlowfreq = dct[:hash_size, 1:hash_size+1]
/// ```
///
/// You can't actually do that with a Python array of arrays... this is numpy
/// 2-D array manipulation magic, where you can index 2-D arrays in slices.
/// Like this:
/// ```ipython3
/// In [0]: x = [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8], [8, 9, 10, 11, 12], [12, 13, 14, 15, 16], [16, 17, 18, 19, 20]]
///
/// In [1]: h = 3
///
/// In [2]: n = np.asarray(x)
///
/// In [3]: lf = n[:h, 1:h+1]
///
/// In [4]: n
/// Out[4]:
/// array([[ 0,  1,  2,  3,  4],
///        [ 4,  5,  6,  7,  8],
///        [ 8,  9, 10, 11, 12],
///        [12, 13, 14, 15, 16],
///        [16, 17, 18, 19, 20]])
///
/// In [5]: lf
/// Out[5]:
/// array([[ 1,  2,  3],
///        [ 5,  6,  7],
///        [ 9, 10, 11]])
/// ```
///
/// Mickey's implementation here was mostly right, except it was including the
/// first column and then later manually excluding the very first value from the
/// sum & average calculation. With a very small tweak, this works correctly to
/// skip the entire first column and include the 8th column.
///
/// The rest of the hash calculation worked correctly. With these changes, this
/// function prints hashes that match imagehash.phash_simple().
///
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

    // Convert the data to a Vec of 64-bit floats.
    let imgbuff_u8 = image_small.into_bytes();
    let mut imgbuff_f64: Vec<f64> = imgbuff_u8.into_iter().map(|x| x as f64).collect();

    // Compute DCT-2 in-place for each row of 32 pixels.
    let dct2 = DctPlanner::new().plan_dct2(32);
    for mut chunk in imgbuff_f64.chunks_mut(32) {
        dct2.process_dct2(&mut chunk);
    }

    // Multiply each value x2, to match results from scipy.fftpack.dct() implementation.
    let _: () = imgbuff_f64
        .iter_mut()
        .map(|f| {
            *f = *f * 2.0;
        })
        .collect();

    // Construct a DCT low frequency vector using the top=left most 8x8 terms,
    // excluding the 0'th column.
    let mut dct_low_freq: Vec<f64> = Vec::new();

    for i in 0..8 {
        let idx = i * 32;
        dct_low_freq.extend_from_slice(&imgbuff_f64[idx + 1..idx + 1 + 8]);
    }

    // Calculate average (mean) of the DCT low frequency vector
    let sum: f64 = dct_low_freq.iter().sum();
    let mean = sum / (dct_low_freq.len() as f64);

    // Construct hash vector by reducing DCT values to 1 or 0 by comparing terms vs mean
    let hashvec: Vec<u64> = dct_low_freq
        .into_iter()
        .map(|x| if x > mean { 1 } else { 0 })
        .collect();

    // Construct hash integer from bits
    let hash: u64 = hashvec.iter().fold(0, |res, &bit| (res << 1) ^ bit);

    if include_filename && debug {
        println!("{:?}: {:#x}", &path, hash);
    } else {
        println!("{:#x}", hash);
    }
    Ok(())
}
