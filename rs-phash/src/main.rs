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

fn print_phash_from_img_path(path: &Path, include_filename: bool, debug: bool) -> Result<()> {

    let image_gs = image::io::Reader::open(&path)
        .map_err(|e| anyhow!("opening {:?}: {}", &path, e))?
        .with_guessed_format()
        .map_err(|e| anyhow!("guessing type {:?}: {}", &path, e))?
        .decode()
        .map_err(|e| anyhow!("decoding {:?}: {}", &path, e))?
        // convert image to grayscale, then resize
        .grayscale();

    let imgbuff_tmp = image::DynamicImage::resize_exact(&image_gs, 32, 32, Lanczos3).to_bytes();
    let mut imgbuff: Vec<f32> = imgbuff_tmp.into_iter().map(|x| x as f32).collect();

    // perform discrete cosine transform (dct2) on image
    DctPlanner::new().plan_dct2(1024).process_dct2(&mut imgbuff);

    // construct a low frequency dct vector using the topmost 8x8 terms
    let mut dct_low_freq: Vec<f32> = Vec::new();

    for i in 0..8 {
        let idx = i * 32;
        dct_low_freq.extend_from_slice(&imgbuff[idx..idx + 8]);
    }

    // take the low frequency averages, excluding the first term
    let sum: f32 = dct_low_freq.iter().sum();
    let sum = sum - dct_low_freq[0];
    let mean = sum / (dct_low_freq.len() - 1) as f32;

    // construct hash vector by reducing DCT values to 1 or 0 by comparing terms vs mean
    let hashvec: Vec<u64> = dct_low_freq
        .into_iter()
        .map(|x| if x > mean { 1 } else { 0 })
        .collect();

    // construct hash integer from bits
    let hash: u64 = hashvec.iter().fold(0, |res, &bit| (res << 1) ^ bit);

    if include_filename && debug {
        println!("{:?}: {:#x}", &path, hash);
    } else {
        println!("{:#x}", hash);
    }
    Ok(())
}