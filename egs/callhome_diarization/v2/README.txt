 This recipe replaces i-vectors used in the v1 recipe with embeddings extracted
 from a deep neural network.  In the scripts, we refer to these embeddings as
 "x-vectors."  The x-vector recipe in
 local/nnet3/xvector/tuning/run_xvector_1a.sh is closesly based on the
 following paper: 

 @inproceedings{snyder2018xvector,
 title={X-vectors: Robust DNN Embeddings for Speaker Recognition},
 author={Snyder, D. and Garcia-Romero, D. and Sell, G. and Povey, D. and Khudanpur, S.},
 booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 year={2018},
 organization={IEEE},
 url={http://www.danielpovey.com/files/2018_icassp_xvectors.pdf}
 }

 However, in this example, the x-vectors are used for diarization, rather
 than speaker recognition.  Diarization is performed by splitting speech
 segments into very short segments (e.g., 1.5 seconds), extracting embeddings
 from the segments, and clustering them to obtain speaker labels.

 The recipe uses the following data for system development.  This is in
 addition to the NIST SRE 2000 dataset (Callhome) which is used for
 evaluation (see ../README.txt).

     Corpus              LDC Catalog No.
     SRE2004             LDC2006S44
     SRE2005 Train       LDC2011S01
     SRE2005 Test        LDC2011S04
     SRE2006 Train       LDC2011S09
     SRE2006 Test 1      LDC2011S10
     SRE2006 Test 2      LDC2012S01
     SRE2008 Train       LDC2011S05
     SRE2008 Test        LDC2011S08
     SWBD2 Phase 2       LDC99S79
     SWBD2 Phase 3       LDC2002S06
     SWBD Cellular 1     LDC2001S13
     SWBD Cellular 2     LDC2004S07

 The following datasets are used in data augmentation.

     MUSAN               http://www.openslr.org/17
     RIR_NOISES          http://www.openslr.org/28
