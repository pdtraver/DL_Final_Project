# DL_Final_Project
Deep Learning (NYU) Final Project \
\
Current Plan (4/9/24 10:54AM): \
    &nbsp;&nbsp;&nbsp;&nbsp;- Get skeleton code for RAFT (by 4/10) \
    &nbsp;&nbsp;&nbsp;&nbsp;- Train RAFT on pairs of labeled data in train set (by 4/12) \
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Split train set into train_train & train_val \
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Treat each video in train_train as mini-batch of 21 pairs of images \
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Train optical flow on train_train set; validate on train_val set \
    &nbsp;&nbsp;&nbsp;&nbsp;- Test trained RAFT on validation set (by 4/14) \
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- With pre-trained RAFT, shift task to predict last frame given first 11 \
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Use validation set as training set; train_val set as validation set (these examples have not been seen in training) \
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- From the 11th frame, predict optical flow of frames 12-22; use last optical flow projection as prediction mask \
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Train on ground truth mask and prediction mask IoU \
\
Dataset stored at /scratch/pdt9929/DL_Final_Project/dataset \
    &nbsp;&nbsp;&nbsp;&nbsp;- Let me know if permissions aren't set right