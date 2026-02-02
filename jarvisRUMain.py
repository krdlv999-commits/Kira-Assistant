import torch
import sounddevice as sd
import time
import datetime
import os
import sys
import webbrowser
import psutil 
import queue
import urllib.parse
import json
import winsound
import random
import threading
import cv2
import base64
import traceback
import pyperclip 
import screen_brightness_control as sbc 
import pyautogui 
import re 
from num2words import num2words 
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume 
from openai import OpenAI
from deep_translator import GoogleTranslator
from rapidfuzz import process, fuzz
from vosk import Model, KaldiRecognizer
from duckduckgo_search import DDGS 
from dotenv import load_dotenv

# --- 1. НАСТРОЙКИ ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("!!! ОШИБКА: Нет ключа в .env !!!")
else:
    print(f"Ключ загружен: {OPENAI_API_KEY[:5]}...")

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
MIC_ID = 1      
CAMERA_ID = 0   

THINKING_PHRASES = ["Смотрю...", "Изучаю...", "Секунду...", "Сейчас проверю...", "Минутку..."]

# --- НАСТРОЙКА МОДЕЛЕЙ (Самая стабильная связка) ---
# 1. Текст: DeepSeek V3 (Умный и дешевый)
CHAT_MODELS = [
    "deepseek/deepseek-chat",  
    "openai/gpt-4o-mini"       
]

# 2. Зрение: GPT-4o-mini (Самая стабильная для картинок)
# Мы убрали Gemini, так как она иногда сбоит через этот API
VISION_MODEL = "openai/gpt-4o-mini"

WAKE_WORDS = ["кира", "киру", "кире", "kira", "юра", "ира"] 

MEMORY_FILE = "memory.json" 
USER_DATA_FILE = "user_data.json" 
CHAT_HISTORY = [] 
q = queue.Queue() 

# --- 2. ИНИЦИАЛИЗАЦИЯ ---
if not os.path.exists("model"): sys.exit("ОШИБКА: Нет папки model!")
vosk_model = Model("model")

print("Загрузка голоса...")
device = torch.device('cpu')
local_file = 'model_silero.pt'
if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt', local_file)  
model_tts = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model_tts.to(device)

sys.stdout.reconfigure(encoding='utf-8')
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENAI_API_KEY)

try: webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(CHROME_PATH))
except: pass

volume_control = None
try:
    from comtypes import CoInitialize
    CoInitialize()
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_control = cast(interface, POINTER(IAudioEndpointVolume))
except: pass

MANUAL_APPS = {
    "калькулятор": "calc.exe", "блокнот": "notepad.exe", "cmd": "cmd.exe",
    "хром": CHROME_PATH, "браузер": CHROME_PATH, "диспетчер": "taskmgr.exe",
    "телеграм": os.path.expandvars(r"%APPDATA%\Telegram Desktop\Telegram.exe")
}
SYSTEM_APPS = {}
CORRECTIONS = {
    "led the": "led zeppelin", "хмм": "хром", "открой хмм": "запусти хром",
    "округе": "в руке", "что у меня округе": "что у меня в руке",
    "путь": "пульт", "это не путь": "это не пульт", "выходк": "выход",
    "а теперь": "что это", "а сейчас": "что это", "посмотри еще раз": "что это",
    "айдар": "а это", "ай да": "а это", "горю": "смотри", 
    "крида": "кира", "что и там": "что это", "что у меня в руки": "что у меня в руке",
    "ира то мне": "кира что мне", "юра": "кира", "луч камеру": "включи камеру", "бюро": "кира"
}

def load_corrections():
    if not os.path.exists(MEMORY_FILE): return CORRECTIONS
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {**CORRECTIONS, **data}
    except: return CORRECTIONS

CORRECTIONS = load_corrections()

# --- 2.1 ПАМЯТЬ ---
def load_user_data():
    if not os.path.exists(USER_DATA_FILE): return {}
    try:
        with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return {}

def save_user_data(key, value):
    data = load_user_data()
    data[key] = value
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

USER_FACTS = load_user_data()

# --- 3. ЗРЕНИЕ ---
class CameraEye:
    def __init__(self):
        self.running = False
        self.cap = None
        self.current_frame = None
        self.current_status = "Камера выключена"
        self.model = None 
        self.translations = {
            "person": "человек", "cell phone": "телефон", "cup": "чашка", "bottle": "бутылка",
            "keyboard": "клавиатура", "mouse": "мышка", "remote": "пульт", "book": "книга",
            "laptop": "ноутбук", "scissors": "ножницы", "pen": "ручка", "spoon": "ложка",
            "baseball bat": "бита/палка", "toothbrush": "зубная щетка"
        }

    def _load_model(self):
        if self.model is None:
            print(">>> Подгружаю YOLO...")
            from ultralytics import YOLO 
            self.model = YOLO("yolov8s.pt")

    def start(self):
        if self.running: return
        self._load_model()
        self.running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.current_status = "Камера выключена"
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

    def get_snapshot_base64(self):
        if not self.running:
            self._load_model()
            cap = cv2.VideoCapture(CAMERA_ID)
            for _ in range(5): ret, frame = cap.read()
            cap.release()
            if not ret: return None
        elif self.current_frame is None:
            return None
        else:
            frame = self.current_frame.copy()
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        return base64.b64encode(buffer).decode('utf-8')

    def _update_loop(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            self.current_frame = frame 
            results = self.model(frame, verbose=False, stream=True, conf=0.45)
            detected_objects = []
            for r in results:
                frame = r.plot()
                for box in r.boxes:
                    obj_name = self.model.names[int(box.cls[0])]
                    detected_objects.append(self.translations.get(obj_name, obj_name))
            if detected_objects:
                counts = {i:detected_objects.count(i) for i in detected_objects}
                self.current_status = ", ".join([f"{k}" if v==1 else f"{k} ({v})" for k,v in counts.items()])
            else: self.current_status = "Пусто"
            cv2.imshow("Kira Vision (YOLO)", frame)
            if cv2.waitKey(1) == ord('q'): self.stop(); break

kira_eye = CameraEye()

def audio_callback(indata, frames, time, status):
    if status: print(status, file=sys.stderr)
    q.put(bytes(indata))

def play_sound(type):
    try:
        if type == "wake": winsound.Beep(1000, 200)
        elif type == "end": winsound.Beep(1500, 100)
        elif type == "sleep": winsound.Beep(700, 300)
    except: pass

def speak(text):
    if not text: return
    print(f"\nKira: {text}") 
    try:
        def replace_numbers(match):
            return num2words(int(match.group()), lang='ru')
        clean_text = re.sub(r'\d+', replace_numbers, text)
        clean_text = clean_text.replace("<s>", "").replace("*", "")
        clean_text = clean_text.replace("#", "").replace("**", "") 
        
        audio = model_tts.apply_tts(text=clean_text, speaker='xenia', sample_rate=48000, put_accent=True, put_yo=True)
        sd.play(torch.cat((audio, torch.zeros(int(48000 * 0.5)))), 48000)
        sd.wait()
    except Exception as e:
        print(f"Ошибка TTS: {e}")

def scan_installed_apps():
    global SYSTEM_APPS
    paths = [os.path.join(os.environ['PROGRAMDATA'], r'Microsoft\Windows\Start Menu\Programs'), os.path.join(os.environ['APPDATA'], r'Microsoft\Windows\Start Menu\Programs')]
    for path in paths:
        if not os.path.exists(path): continue
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".lnk"):
                    SYSTEM_APPS[file.lower().replace(".lnk", "").replace("  ", " ")] = os.path.join(root, file)

def open_in_chrome(url):
    try: webbrowser.get('chrome').open(url)
    except: webbrowser.open(url)

def search_internet(query):
    print(f"🌍 Ищу в интернете: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except: pass
    return None

# --- МОЗГ ---
def ask_gpt_text(prompt):
    global CHAT_HISTORY, USER_FACTS
    visual_context = kira_eye.current_status
    user_context = f"Факты о пользователе: {json.dumps(USER_FACTS, ensure_ascii=False)}" if USER_FACTS else ""
    
    now = datetime.datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M")
    
    print(f"Думаю... (Вижу: {visual_context})", end='\r')
    
    system_prompt = (
        f"Ты Кира. СЕГОДНЯ: {date_str}, ВРЕМЯ: {time_str}. "
        f"Твои глаза видят: {visual_context}. {user_context}. "
        f"Если пользователь просит ЗАПОМНИТЬ, верни: MEMORY: <ключ>|<значение>. "
        f"Если спрашивает новости/погоду (и ты не знаешь), верни: SEARCH: <запрос>. "
        f"Отвечай кратко на русском."
    )

    for model in CHAT_MODELS:
        try:
            messages = [{"role": "system", "content": system_prompt}] + CHAT_HISTORY + [{"role": "user", "content": prompt}]
            
            completion = client.chat.completions.create(model=model, messages=messages, max_tokens=300)
            answer = completion.choices[0].message.content
            
            if "SEARCH:" in answer:
                search_query = answer.replace("SEARCH:", "").strip()
                search_results = search_internet(search_query)
                if search_results:
                    follow_up = f"Результаты поиска: {search_results}\nСегодня {date_str}. Ответь кратко."
                    messages.append({"role": "assistant", "content": answer}) 
                    messages.append({"role": "user", "content": follow_up})
                    completion = client.chat.completions.create(model=model, messages=messages, max_tokens=300)
                    answer = completion.choices[0].message.content
                else: answer = "Ничего не нашла в интернете."

            elif "MEMORY:" in answer:
                try:
                    raw = answer.replace("MEMORY:", "").strip()
                    key, value = raw.split("|", 1)
                    save_user_data(key.strip(), value.strip())
                    USER_FACTS = load_user_data()
                    answer = f"Запомнила: {key} - {value}"
                except: answer = "Не удалось сохранить в память."

            try: answer = GoogleTranslator(source='auto', target='ru').translate(answer)
            except: pass
            
            CHAT_HISTORY.append({"role": "user", "content": prompt})
            CHAT_HISTORY.append({"role": "assistant", "content": answer})
            if len(CHAT_HISTORY) > 10: CHAT_HISTORY = CHAT_HISTORY[-10:]
            return answer
        except Exception as e:
            print(f"\nОшибка модели {model}: {e}")
            continue
            
    return "Мозг оффлайн (все модели недоступны)."

def ask_gpt_vision(prompt):
    global CHAT_HISTORY
    print("Делаю снимок...", end='\r')
    base64_image = kira_eye.get_snapshot_base64()
    if not base64_image: return "Не вижу изображения."
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Посмотри и ответь: {prompt}. Отвечай кратко, по-русски."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
        # ДЛЯ ЗРЕНИЯ ИСПОЛЬЗУЕМ GPT-4o-mini (Стабильность 100%)
        completion = client.chat.completions.create(model=VISION_MODEL, messages=messages, max_tokens=300)
        answer = completion.choices[0].message.content
        CHAT_HISTORY.append({"role": "user", "content": f"[Фото: {prompt}]"})
        CHAT_HISTORY.append({"role": "assistant", "content": f"[На фото: {answer}]"})
        return answer
    except Exception as e:
        print(f"\nОШИБКА ЗРЕНИЯ: {e}") 
        return "Зрение недоступно."

# --- КОМАНДЫ ---
def execute_command(query):
    if not query: return
    print(f"\nКоманда: {query}")
    play_sound("end")
    for w, r in CORRECTIONS.items():
        if w in query: query = query.replace(w, r)
    if any(x in query for x in ['отбой', 'хватит', 'стоп', 'замолчи']):
         speak("Ок.")
         return "SLEEP_NOW" 
    
    if 'буфер' in query or 'скопирован' in query:
        if 'исправь' in query or 'проверь' in query or 'отредактируй' in query:
            text_to_fix = pyperclip.paste()
            if not text_to_fix: speak("Буфер пуст."); return
            speak("Исправляю...")
            prompt = f"Исправь ошибки и улучши: {text_to_fix}"
            fixed_text = ask_gpt_text(prompt)
            pyperclip.copy(fixed_text)
            speak("Готово.")
            return

    if 'пауза' in query or 'продолж' in query: pyautogui.press('playpause'); return
    if 'следующий трек' in query: pyautogui.press('nexttrack'); return
    if 'предыдущий трек' in query: pyautogui.press('prevtrack'); return
    if 'сверни все' in query: pyautogui.hotkey('win', 'd'); return

    vision_triggers = ["что это", "что я держу", "что в руке", "опиши", "посмотри", "что ты видишь", "а это", "скажи что это", "какого цвета"]
    is_vision = False
    for t in vision_triggers:
        if fuzz.partial_ratio(t, query) > 85: is_vision = True; break     
    if is_vision:
        if not kira_eye.running: 
            kira_eye.start()
            speak("Включаю глаза...")
            time.sleep(2) 
        speak(random.choice(THINKING_PHRASES))
        description = ask_gpt_vision(query)
        speak(description)
        return

    if 'яркость' in query:
        try:
            current = sbc.get_brightness()[0]
            if any(x in query for x in ['прибавь', 'увеличь', 'добавь', 'больше']): sbc.set_brightness(min(100, current + 20)); speak("Ярче.")
            elif any(x in query for x in ['убавь', 'уменьши', 'тише', 'меньше']): sbc.set_brightness(max(0, current - 20)); speak("Темнее.")
        except: speak("Не могу.")
        return
    if 'громкость' in query or 'звук' in query:
        if volume_control:
            if any(x in query for x in ['громче', 'добавь']):
                c = volume_control.GetMasterVolumeLevelScalar()
                volume_control.SetMasterVolumeLevelScalar(min(1.0, c + 0.1), None)
                speak("Громче.")
            elif any(x in query for x in ['тише', 'убавь']):
                c = volume_control.GetMasterVolumeLevelScalar()
                volume_control.SetMasterVolumeLevelScalar(max(0.0, c - 0.1), None)
                speak("Тише.")
            elif 'выключи' in query: volume_control.SetMute(1, None); speak("Выключила.")
            elif 'включи' in query: volume_control.SetMute(0, None); speak("Включила.")
        return
    if 'заряд' in query:
        b = psutil.sensors_battery()
        if b: speak(f"Заряд {b.percent}%.")
        else: speak("Нет батареи.")
        return

    if 'включи камеру' in query: kira_eye.start(); speak("Глаза открыты."); return
    if 'выключи камеру' in query: kira_eye.stop(); speak("Глаза закрыты."); return
    if 'кто ты' in query: speak("Я Кира."); return
    if fuzz.partial_ratio("сколько времени", query) > 75: speak(f"Сейчас {datetime.datetime.now().strftime('%H:%M')}"); return
    if 'юту' in query:
        search_term = query.replace('включи', '').replace('найди', '').replace('ютуб', '').replace('юту', '').strip()
        encoded_term = urllib.parse.quote(search_term)
        open_in_chrome(f"https://www.youtube.com/results?search_query={encoded_term}")
        return
    if 'запусти' in query or 'открой' in query:
        raw_app_name = query.replace('запусти', '').replace('открой', '').replace('программу', '').strip()
        if raw_app_name:
            speak(f"Ищу {raw_app_name}...") 
            ALL_APPS = {**MANUAL_APPS, **SYSTEM_APPS}
            result = process.extractOne(raw_app_name, ALL_APPS.keys(), scorer=fuzz.ratio, score_cutoff=60)
            if result:
                speak(f"Запускаю {result[0]}")
                try: os.startfile(ALL_APPS[result[0]])
                except: speak("Ошибка файла.")
            else: speak(f"Не нашла {raw_app_name}")
        return
    if 'выход' in query: kira_eye.stop(); speak("Пока."); sys.exit()
    speak(ask_gpt_text(query))

def main():
    scan_installed_apps()
    rec = KaldiRecognizer(vosk_model, 16000)
    print(f"Подключаюсь к микрофону ID {MIC_ID}...")
    try:
        input_stream = sd.RawInputStream(samplerate=16000, blocksize=8000, device=MIC_ID, 
                                         dtype='int16', channels=1, callback=audio_callback)
    except Exception as e:
        print(f"ОШИБКА МИКРОФОНА: {e}"); sys.exit()

    speak("Я на связи. Jarvis 47.0")
    dialogue_mode = True 
    last_interaction_time = time.time()
    
    print(">>> СЛУШАЮ (30 сек) <<<")
    
    with input_stream:
        while True:
            try:
                data = q.get() 
                status_text = kira_eye.current_status[:40] if kira_eye.running else "OFF"
                print(f"YOLO: {status_text}... | Слушаю... ", end='\r')
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get('text', '')

                    if text:
                        print(" " * 80, end='\r') 
                        if not dialogue_mode:
                            if any(name in text for name in WAKE_WORDS):
                                print("\n>>> ПРОСНУЛАСЬ <<<")
                                play_sound("wake")
                                dialogue_mode = True
                                last_interaction_time = time.time()
                                for name in WAKE_WORDS:
                                    if name in text:
                                        command = text.replace(name, "").strip()
                                        if len(command) > 2:
                                            res = execute_command(command)
                                            if res == "SLEEP_NOW": dialogue_mode = False
                                            else: last_interaction_time = time.time()
                                        break
                        else:
                            print(f"Слышу: {text}")
                            res = execute_command(text)
                            if res == "SLEEP_NOW": 
                                print(">>> СОН <<<")
                                play_sound("sleep")
                                dialogue_mode = False
                            else: 
                                last_interaction_time = time.time()
                
                if dialogue_mode and (time.time() - last_interaction_time > 30):
                    print("\n>>> ТАЙМАУТ: УСНУЛА <<<")
                    play_sound("sleep")
                    dialogue_mode = False
            except KeyboardInterrupt: break
            except Exception as e: print(f"\nОшибка: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main()