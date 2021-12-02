CONFIG_FILE='configs/endometrial_lagenet.yaml'
WOKERS=16

python s1_cnn_sample.py --cfg $CONFIG_FILE --num-workers $WOKERS

for((FOLD=0;FOLD<5;FOLD++)); 
do
    python s2_cnn_train.py --cfg $CONFIG_FILE --epochs 50 --fold $FOLD\
        --batch-size 256 -j $WOKERS --weighted-sample\
        --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0

    python s3_cnn_wsi_encode.py --cfg $CONFIG_FILE --num-workers $WOKERS --batch-size 512\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
    
    python s4_graph_sample.py --cfg $CONFIG_FILE --num-workers $WOKERS --fold $FOLD

    python s5_lagenet_train.py --cfg $CONFIG_FILE --gpu 0 --num-workers $WOKERS --fold $FOLD\
          --num-epochs 300 --batch-size 64 --redo
done
