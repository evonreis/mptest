use std::ops::Deref;
use std::sync::Arc;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use ndarray::parallel::prelude::*;
use rand::Rng;
use ndarray::{Array,ArrayView, Dim};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;
use std::vec;
use rustfft::num_complex::Complex64;

const NUM_TFS: usize = 48*48;
const TIME_S: usize = 20;
const RATE_HZ: usize = 16384;
const SAMPLES: usize = TIME_S*RATE_HZ;

fn do_fft(fft: &Arc<dyn Fft<f64>>, in_data: &ArrayView<f64, Dim<[usize; 1]>>) -> Vec<Complex64> {
    let mut in_data_vec: Vec<Complex64> = in_data.iter()
        .map(|&i| Complex64::new(i, 0.0))
        .collect();

    //fft.process(&mut in_data_vec);
    return in_data_vec;
}

fn main() {


    println!("Creating inputs");
    let time_data =
        Array::random((NUM_TFS, SAMPLES), Uniform::new(0., 1.));

    let thread_counts: [i32; 4] = [1,2,4,8];

    let mut total: usize = 0;

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = planner.plan_fft_forward(SAMPLES);

    for threads in thread_counts.iter() {
        let mut total: usize = 0;
        println!("timing threads");
        let now = Instant::now();
        let results = time_data.outer_iter().into_par_iter()
            .map(|i| do_fft(&fft, &i));
        total += results.count();
        let elapsed_time = now.elapsed();
        println!("Running fft threads = {} took {} seconds to do {} calcs.",
                 threads, elapsed_time.as_secs(), total);
    }

    println!("Hello, world!");
}
