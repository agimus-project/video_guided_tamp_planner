# Multi-Contact Task and Motion Planning Guided by Video Demonstration


## Installation
Please follow the installation procedure of GuidedTAMPBenchmark [repo](https://github.com/agimus-project/guided_tamp_benchmark). 
```bash
conda config --add channels conda-forge
conda activate gtamp  # activate your GuidedTAMPBenchmark environment
conda install hpp-gepetto-viewer -y  # installs HPP libraries
conda install pycollada anytree quaternion  # additional planner dependencies
```

## Data preprocessing
To run the planning there are two possibilities:
1. Run the algorithm on data files from GuidedTAMPBenchmark repo (no data preprocessing needed)
2. Replace the GuidedTAMPBenchmark data files with your custom data

The steps to create the data files are the following:
- shoot the video where human hands and objects are clearly visible at times of contact
- process the video with 6D pose estimator (f.e. [CosyPose](https://github.com/ylabbe/cosypose))
- process the video with hand-object contact recognizer (f.e. [Hand Object Detector](https://github.com/ddshan/hand_object_detector))
- follow the [instructions](preprocessing/doc.md) in the preprocessing folder

## Run the code
To run the benchmark script, use the following commands:
```bash
conda activate gtamp
export PYTHONPATH=$PYTHONPATH:`pwd`
python scripts/01_benchmark.py -task_name shelf -task_id 1 -robot_name panda -planner multi_contact -res_file data/res.pkl
python scripts/01_benchmark.py -task_name shelf -task_id 1 -robot_name ur5 -planner multi_contact -res_file data/res.pkl
```

To run for a single task and visualise the policy, use the file `script/02_solve_task.py`

To visualize the quantitative results, run:
```bash
python visualisations/paper_visualization.py -res_file data/res.pkl
```
#### Install PDDLStream

In order to run `PDDL` method in benchmark, follow next steps:
- Install PDDLStream [repo](https://github.com/caelan/pddlstream)
  - add import of `MutableSet` to `pddlstream/examples/pybullet/utils/pybullet_tools/utils.py`
  - if you use python >= 3.10, fix `Sequence`, `Iterator` and `Sized` imports from 
    collection in 
    `pddlstream/language/generator.py`, `pddlstream/language/stream.py`, 
    `pddlstream/algorithms/instantiation.py`, `pddlstream/algorithms/skeleton.py`
- Pass the correct `-pddl_path` argument to scripts
