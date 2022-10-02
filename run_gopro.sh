# ROOT_DATASETS=/media/Data/Sparsenet/TestAlignment/

# FILENAME=bike1_trail1_linear
# IMG_DIR=${ROOT_DATASETS}/${FILENAME}_imgs
# SAVE_DIR=${ROOT_DATASETS}/dpvo_result_${FILENAME}

# python demo.py --imagedir ${IMG_DIR} \
# --calib calib/gopro9_linear.txt --config config/medium.yaml \
# --savefile ${SAVE_DIR} --save_mapfile ${SAVE_DIR}_map \
# --start_t_ns 68000000000 --end_t_ns 85580000000 --stride 1 

# FILENAME=bike2_trail1_linear
# IMG_DIR=${ROOT_DATASETS}/${FILENAME}_imgs
# SAVE_DIR=${ROOT_DATASETS}/dpvo_result_${FILENAME}

# python demo.py --imagedir ${IMG_DIR} \
# --calib calib/gopro9_linear.txt --config config/medium.yaml \
# --savefile ${SAVE_DIR} --save_mapfile ${SAVE_DIR}_map \
# --start_t_ns 30180000000 --end_t_ns 48180000000 --stride 1 


ROOT_DATASETS=/media/Data/Sparsenet/Ammerbach/Links

FILENAME=run1
IMG_DIR=${ROOT_DATASETS}/${FILENAME}
SAVE_DIR=${ROOT_DATASETS}/dpvo_result_${FILENAME}

python demo.py --imagedir ${IMG_DIR} \
--calib calib/gopro9_1440_linear.txt --config config/medium.yaml \
--savefile ${SAVE_DIR} --save_mapfile ${SAVE_DIR}_map \
--start_t_ns 3140000000 --end_t_ns 146860000000 --stride 1

FILENAME=run2
IMG_DIR=${ROOT_DATASETS}/${FILENAME}
SAVE_DIR=${ROOT_DATASETS}/dpvo_result_${FILENAME}

python demo.py --imagedir ${IMG_DIR} \
--calib calib/gopro9_1440_linear.txt --config config/medium.yaml \
--savefile ${SAVE_DIR} --save_mapfile ${SAVE_DIR}_map \
--start_t_ns 2300000000 --end_t_ns 138620000000 --stride 1 

ROOT_DATASETS=/media/Data/Sparsenet/Ammerbach/Links

# FILENAME=run1
# IMG_DIR=${ROOT_DATASETS}/${FILENAME}
# SAVE_DIR=${ROOT_DATASETS}/dpvo_result_${FILENAME}

# python demo.py --imagedir ${IMG_DIR} \
# --calib calib/gopro9_1440_linear.txt --config config/medium.yaml \
# --savefile ${SAVE_DIR} --save_mapfile ${SAVE_DIR}_map \
# --start_t_ns 3140000000 --end_t_ns 11700000000 --stride 1

# FILENAME=run2
# IMG_DIR=${ROOT_DATASETS}/${FILENAME}
# SAVE_DIR=${ROOT_DATASETS}/dpvo_result_${FILENAME}

# python demo.py --imagedir ${IMG_DIR} \
# --calib calib/gopro9_1440_linear.txt --config config/medium.yaml \
# --savefile ${SAVE_DIR} --save_mapfile ${SAVE_DIR}_map \
# --start_t_ns 2300000000 --end_t_ns 10580000000 --stride 1 