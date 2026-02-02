import pyaudio

p = pyaudio.PyAudio()

print("\n--- СПИСОК МИКРОФОНОВ ---")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0: # Только устройства ввода
        print(f"ID: {i}  -  {info['name']}")

print("-------------------------\n")
input("Нажми Enter для выхода...")