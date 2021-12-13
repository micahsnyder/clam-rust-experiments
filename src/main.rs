extern crate image;

use image::imageops::FilterType::Lanczos3;
use rustdct::DctPlanner;

use std::env;
use std::process;
fn main() {

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        process::exit(2);
    }

    for i in 1..args.len() {
        let path = &args[i];
        print_phash_from_img_path(path);
    }
}

fn print_phash_from_img_path(path: &String) { 
    let res = image::open(path);
    let image1;
    match res {
        Ok(img) => image1 = img,
        Err(error) => {
            println!("error decoding image... path: {:?}, err: {:?}", path, error);
            return ();
        },
    };

    // convert image to grayscale, then resize
    let image1 = image1.grayscale();
    let imgbuff_tmp = image::DynamicImage::resize_exact(&image1, 32, 32, Lanczos3).to_bytes();
    let mut imgbuff: Vec<f32> = imgbuff_tmp.into_iter().map(|x| x as f32).collect();

    // perform discrete cosine transform (dct2) on image 
    let mut planner= DctPlanner::new(); 
    let dct2 = planner.plan_dct2(1024);

    dct2.process_dct2(&mut imgbuff);


    // construct a low frequency dct vector using the topmost 8x8 terms
    let mut dct_low_freq: Vec<f32> = Vec::new();

    for i in 0..8 {
        let idx = i * 32;
        dct_low_freq.extend_from_slice(&imgbuff[idx..idx+8]);
    }

    // take the low frequency averages, exlcuding the first term
    let sum: f32= dct_low_freq.iter().sum();
    let sum = sum - dct_low_freq[0];
    let mean = sum / (dct_low_freq.len() - 1) as f32;

    // construct hash vector by reducing DCT values to 1 or 0 by comparing terms vs mean
    let hashvec: Vec<u64> = dct_low_freq.into_iter()
                                        .map(|x| 
                                            if x > mean {1} else {0}
                                        ).collect();

    // construct hash integer from bits
    let hash: u64 = hashvec.iter()
                            .fold(0, |res, &bit| { 
                                (res << 1) ^ bit} 
                            );
    
    println!("{:#x}", hash);    
}