import sys, os

filename = sys.argv[1]
data_dir = sys.argv[2]
if not os.path.exists(data_dir):
  os.makedirs(data_dir)
wavscp_fi = open(data_dir + "/wav.scp" , 'w')
utt2spk_fi = open(data_dir + "/utt2spk" , 'w')
utt, ext = os.path.basename(filename).split(".")
if ext == "wav":
  spkr = utt
  wav_str = utt + " " + filename + "\n"
  wavscp_fi.write(wav_str)
  utt2spk_str = utt + " " + spkr + "\n"
  utt2spk_fi.write(utt2spk_str)

utt2spk_fi.close()
wavscp_fi.close()
