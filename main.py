import soundfile
from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader

# d = ModelDownloader("./model/pretrained-model")  # Specify cachedir
# d.query( task="asr", corpus='librispeech')
# d.download_and_unpack("kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best@<revision>")
# d.download_and_unpack("kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best")
# d.download_and_unpack(task="asr", corpus="wsj")
# d.download_and_unpack(task="asr", corpus="wsj", version=-1)  # Get the last model
# d.download_and_unpack("https://zenodo.org/record/...")
# d.download_and_unpack("./some/where/model.zip")

if __name__ == '__main__':
    from espnet2.bin.asr_inference import Speech2Text
    
    model = Speech2Text.from_pretrained(
        "espnet/simpleoier_librispeech_asr_train_asr_conformer7_hubert_ll60k_large_raw_en_bpe5000_sp"
    )
    
    speech, rate = soundfile.read("speech.wav")
    text, *_ = model(speech)
    # d = ModelDownloader(cachedir="./model/pretrained-model")
    # d.download_and_unpack(
    #     'espnet/xuankai_chang_librispeech_asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp_25epoch')
    
    # config = {
    #     'asr_model_file'  : '/home/swang/project/SmartSpeaker/ASR/model/pretrained-model/espnet--xuankai_chang_librispeech_asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp_25epoch.main.19ad373979f4c0b142fd07f6f38f57a57ce5659b/exp_wav2vec2_960hr_large_weighted_perturb/asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp/25epoch.pth',
    #     'lm_file'         : '/home/swang/project/SmartSpeaker/ASR/model/pretrained-model/espnet--xuankai_chang_librispeech_asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp_25epoch.main.19ad373979f4c0b142fd07f6f38f57a57ce5659b/exp_wav2vec2_960hr_large_weighted_perturb/lm_train_lm_transformer2_en_bpe5000/17epoch.pth',
    #     'asr_train_config': '/home/swang/project/SmartSpeaker/ASR/model/pretrained-model/espnet--xuankai_chang_librispeech_asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp_25epoch.main.19ad373979f4c0b142fd07f6f38f57a57ce5659b/exp_wav2vec2_960hr_large_weighted_perturb/asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp/config.yaml',
    #     'lm_train_config' : '/home/swang/project/SmartSpeaker/ASR/model/pretrained-model/espnet--xuankai_chang_librispeech_asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp_25epoch.main.19ad373979f4c0b142fd07f6f38f57a57ce5659b/exp_wav2vec2_960hr_large_weighted_perturb/lm_train_lm_transformer2_en_bpe5000/config.yaml',
    # }
    # model = Speech2Text(**config    )
    #
    # speech, rate = soundfile.read("speech.wav")
    # text, *_ = model(speech)
    #
    # speech2text = Speech2Text.from_pretrained(
    #     "model_name",
    #     # Decoding parameters are not included in the model file
    #     maxlenratio=0.0,
    #     minlenratio=0.0,
    #     beam_size=20,
    #     ctc_weight=0.3,
    #     lm_weight=0.5,
    #     penalty=0.0,
    #     nbest=1
    # )
    # # Confirm the sampling rate is equal to that of the training corpus.
    # # If not, you need to resample the audio data before inputting to speech2text
    # speech, rate = soundfile.read("speech.wav")
    # nbests = speech2text(speech)
    #
    # text, *_ = nbests[0]
    # print(text)
