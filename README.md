```bash
git clone https://github.com/afrizalhasbi/vits
cd vits
uv pip install -r requirements.txt
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
python convert_original_discriminator_checkpoint.py --language_code ind --pytorch_dump_folder_path mms_ind
```
```
accelerate launch run.py config.json
```

## Inference

You can use a finetuned model via the Text-to-Speech (TTS) [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline) in just a few lines of code!
Just replace `ylacombe/vits_ljs_welsh_female_monospeaker_2` with your own model id (`hub_model_id`) or path to the model (`output_dir`).

```python
from transformers import pipeline
import scipy

model_id = "ylacombe/vits_ljs_welsh_female_monospeaker_2"
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

speech = synthesiser("Hello, my dog is cooler than you!")

scipy.io.wavfile.write("finetuned_output.wav", rate=speech["sampling_rate"], data=speech["audio"][0])
```

Note that if your model needs to use `uroman` to train, you also should apply the uroman package to your text inputs prior to passing them to the pipeline:

```python
import os
import subprocess
from transformers import pipeline
import scipy

model_id = "facebook/mms-tts-kor"
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

def uromanize(input_string, uroman_path):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Execute the perl command
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    # Return the output as a string and skip the new-line character at the end
    return stdout.decode()[:-1]

text = "이봐 무슨 일이야"
uromanized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

speech = synthesiser(uromanized_text)

scipy.io.wavfile.write("finetuned_output.wav", rate=speech["sampling_rate"], data=speech["audio"][0])
```

-----------------------------



## Acknowledgements


* [VITS](https://huggingface.co/docs/transformers/model_doc/vits) was proposed in 2021, in [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) by Jaehyeon Kim, Jungil Kong, Juhee Son. You can find the original codebase [here](https://github.com/jaywalnut310/vits).
* [MMS](https://huggingface.co/facebook/mms-tts) was proposed in [Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516) by Vineel Pratap, Andros Tjandra, Bowen Shi and co. You can find more details about the supported languages and their ISO 639-3 codes in the [MMS Language Coverage Overview](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html),
and see all MMS-TTS checkpoints on the Hugging Face Hub: [facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts).
* [Hugging Face 🤗 Transformers](https://huggingface.co/docs/transformers/index) for the model integration, [Hugging Face 🤗 Accelerate](https://huggingface.co/docs/accelerate/index) for the distributed code and [Hugging Face 🤗 datasets](https://huggingface.co/docs/datasets/index) for facilitating datasets access.
* @nivibilla's [adapation](https://github.com/nivibilla/efficient-vits-finetuning) of HifiGan's discriminator, used for English VITS training.
