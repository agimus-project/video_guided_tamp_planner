Instruction

Preprocess video
- convert video to frames and save to a folder
- process the frames with 6D pose estimator (f.e. [CosyPose](https://github.com/ylabbe/cosypose)), save two data files:
  - preprocessing/data/cosypose/{task_name}_{task_id}.pkl which contains 
    dictionary where keys are the object ids and values are N x 4 x 4 arrays which 
    contain object poses (for N frames)
  - preprocessing/data/cosypose/{task_name}_{task_id}_bb_list.pkl which contains 
    dictionary where keys are the object ids and values are N x 4 arrays which 
    contain object bounding box (for N frames)
- process the frames with hand-object contact recognizer (f.e. [Hand Object Detector](https://github.com/ddshan/hand_object_detector)), save two data files:
  - preprocessing/data/handobj/{task_name}_{task_id}.pkl which contains 
    dictionary where keys are the frame ids and values 1 x 10 arrays of hand 
    detection output (or None)
  - preprocessing/data/handobj/{task_name}_{task_id}_obj_dets.pkl wwhich contains 
    dictionary where keys are the frame ids and values 1 x 10 arrays of objects 
    detection output (or None)

Additional instruction to estimate world-object transformation:
- shoot "calibration" video with the same camera pose as in the main video
- place a known object (we used Cheez-it box) at a known pose (f.e. edge of the table) 
- create a .npy file with transformation (in 'world' frame of reference) to the known 
  pose (4 x 4 world_object matrix)
- process the frames with 6D pose estimator, save to calibration.pkl file 
  (dictionary where key is the id on the known object and value is N x 4 x 4 arrays 
  which contain object poses (for N frames))
- run [01_create_calib_matrix.py](01_create_calib_matrix.py) with prober calibration 
  path and world_object matrix

Create demo file:
- run [02_create_data_file.py](02_create_data_file.py)

Sample robot poses
- delete guiding files from `preprocessing/data` folder
- run `for run in {1..100}; do python preprocessing/03_sample_one_pose.py -task_name 
  shelf -task_id 0 -robot ur5 -pose_id 0 -seed $run --verbose; done`


After all steps are done, you can verify the results by placing the demo files and 
robot pose files into `guided_tamp_benchmark/tasks/data` folder in 
GuidedTAMPBenchmark repo and running `examples/02_render_demonstration.py`. 

Useful ffmpeg commands:
1) trim : `ffmpeg -ss 00:00:02 -t 00:00:14 -i v1.mp4 -vcodec copy -acodec copy short_v1.mp4`