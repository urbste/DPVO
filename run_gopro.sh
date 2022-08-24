
FILENAME=bike1_trail1_linear

python demo.py --imagedir /media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/${FILENAME}.MP4 \
--calib calib/gopro9_linear.txt --config config/medium.yaml --savefile /media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/dpvo_result_${FILENAME} \
--start_t 67.0 --end_t 0