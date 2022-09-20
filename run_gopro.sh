ROOT_DATASETS=/media/Data/Sparsenet/TestAlignment/

FILENAME=bike1_trail1_linear
IMG_DIR=${ROOT_DATASETS}/${FILENAME}_imgs
SAVE_DIR=${ROOT_DATASETS}/dpvo_result_${FILENAME}

python demo.py --imagedir ${IMG_DIR} \
--calib calib/gopro9_linear.txt --config config/medium.yaml \
--savefile ${SAVE_DIR} --save_mapfile ${SAVE_DIR}_map \
--start_t_ns 68000000000 --end_t_ns 85580000000 --stride 1 

FILENAME=bike2_trail1_linear
IMG_DIR=${ROOT_DATASETS}/${FILENAME}_imgs
SAVE_DIR=${ROOT_DATASETS}/dpvo_result_${FILENAME}

python demo.py --imagedir ${IMG_DIR} \
--calib calib/gopro9_linear.txt --config config/medium.yaml \
--savefile ${SAVE_DIR} --save_mapfile ${SAVE_DIR}_map \
--start_t_ns 30180000000 --end_t_ns 48180000000 --stride 1 