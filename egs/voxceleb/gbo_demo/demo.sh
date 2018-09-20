#!/bin/bash
. ./cmd.sh
. ./path.sh
train_cmd=run.pl
stage=15
if [ $stage -le 0 ]; then
./create_test_embedding.sh https://www.youtube.com/watch?v=0thYSeNRqi0
fi
if [ $stage -le 1 ]; then
  echo "Testing Mark_Hamill";
 ./verify.sh Mark_Hamill
fi
if [ $stage -le 2 ]; then
  echo "Testing Aaron_Ashmore";
 ./verify.sh Aaron_Ashmore
fi
if [ $stage -le 3 ]; then
  echo "Testing Abu_Qatada";
 ./verify.sh Abu_Qatada
fi
if [ $stage -le 4 ]; then
  echo "Testing Yvette_Nicole_Brown";
 ./verify.sh Yvette_Nicole_Brown
fi
if [ $stage -le 5 ]; then
  echo "Testing Wilson_Cruz";
 ./verify.sh Wilson_Cruz
fi
if [ $stage -le 6 ]; then
  echo "Testing Zac_Efron";
 ./verify.sh Zac_Efron
fi
if [ $stage -le 7 ]; then
  sleep 5
  echo "Getting enrolled speakers"; 
  ./get_enrolled_speakers.sh > enrolled_speakers
fi
if [ $stage -le 8 ]; then
  echo "Performing a voice search";
  ./voice_search.sh > Mark_Hamill_voice_search
fi
if [ $stage -le 9 ]; then
time ./create_test_embedding.sh https://www.youtube.com/watch?v=OpOC4O2jgMs
fi
if [ $stage -le 10 ]; then
  echo "Testing Mark Hamill";
 ./verify.sh Mark_Hamill
fi
if [ $stage -le 11 ]; then
  echo "Testing Catherine_Tyldesley";
 ./verify.sh Catherine_Tyldesley
fi
if [ $stage -le 12 ]; then
  echo "Testing Aaron_Tippin";
 ./verify.sh Aaron_Tippin
fi
if [ $stage -le 13 ]; then
  echo "Testing Joss_Whedon";
 ./verify.sh Joss_Whedon
fi
if [ $stage -le 14 ]; then
  echo "Performing a voice search";
  ./voice_search.sh > Joss_Whedon_voice_search
fi
if [ $stage -le 15 ]; then
time ./create_test_embedding.sh https://www.youtube.com/watch?v=M_E3o2qChxU
fi
if [ $stage -le 16 ]; then
  ./create_enroll_embedding.sh https://www.youtube.com/watch?v=IRkxSRYBWug Sanjeev_Khudanpur
  ./make_enrollments.sh
fi
if [ $stage -le 17 ]; then
  echo "Testing Joss_Whedon";
 ./verify.sh Joss_Whedon
fi
if [ $stage -le 18 ]; then
  echo "Testing Sanjeev_Khudanpur";
 ./verify.sh Sanjeev_Khudanpur
fi
if [ $stage -le 19 ]; then
  echo "Performing a voice search";
  ./voice_search.sh > Sanjeev_Khudanpur_voice_search1
fi

exit 1;
if [ $stage -le 20 ]; then
  ./create_enroll_embedding.sh https://www.youtube.com/watch?v=9YjnVK58RxM Jason_Eisner
  ./make_enrollments.sh
fi
if [ $stage -le 21 ]; then
time ./create_test_embedding.sh https://www.youtube.com/watch?v=e_Rm3TALMsQ
fi
if [ $stage -le 22 ]; then
  echo "Testing Sanjeev_Khudanpur";
 ./verify.sh Sanjeev_Khudanpur
fi
if [ $stage -le 23 ]; then
  echo "Testing Jason_Eisner";
 ./verify.sh Jason_Eisner
fi
if [ $stage -le 24 ]; then
  echo "Performing a voice search";
  ./voice_search.sh > Jason_Eisner_voice_search1
fi
if [ $stage -le 25 ]; then
  ./remove_enrollment.sh Sanjeev_Khudanpur
  ./make_enrollments.sh
fi
if [ $stage -le 26 ]; then
  echo "Performing a voice search";
  ./voice_search.sh > Jason_Eisner_voice_search2
fi
if [ $stage -le 27 ]; then
  ./remove_enrollment.sh Jason_Eisner
  ./make_enrollments.sh
fi
if [ $stage -le 28 ]; then
  echo "Performing a voice search";
  ./voice_search.sh > Jason_Eisner_voice_search3
fi
