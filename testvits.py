from transformers import pipeline
import scipy
from tqdm import tqdm
import os

os.makedirs("contohs", exist_ok=True)
model_id = "output-ft"
synthesiser = pipeline("text-to-speech", model_id, device='cuda')

texts = [
    'Sepanjang jalan kenangan. Kita selalu bergandeng tangan',
    'Dengan senang hati, bapak. Saya senang bisa membantu',
    'Mohon maaf, saya izin interupsi',
    'Apa kabar? Selamat siang! Ada yang bisa saya bantu?',
    'Saya suka tempe, tapi tidak suka tahu',
    'Untuk produk itu tidak kami layani, ibu',
    'Saya harus berikan informasi ini segera kepada anda dan mereka semua.',
    'Kucing itu runcing kuku kaki cakarnya.',
    'Nah, itu yang saya maksud dari kemarin.',
    'Kalau kata atasan saya, request bapak itu melanggar prosedur.'
    ]

for idx, t in tqdm(enumerate(texts)):
    speech = synthesiser(t)
    scipy.io.wavfile.write(f"contohs/contoh_{idx}.wav", rate=speech["sampling_rate"], data=speech["audio"][0])

