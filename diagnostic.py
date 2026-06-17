import sys
import subprocess

print("=" * 50)
print("ДИАГНОСТИКА PYTHON")
print("=" * 50)

print(f"Путь к интерпретатору: {sys.executable}")
print(f"Версия Python: {sys.version}")
print(f"Путь к проекту: {sys.path}")

print("\n" + "=" * 50)
print("ПРОВЕРКА УСТАНОВЛЕННЫХ ПАКЕТОВ")
print("=" * 50)

# Проверяем через pip
result = subprocess.run([sys.executable, "-m", "pip", "list"],
                        capture_output=True, text=True, encoding='utf-8')
print(result.stdout[:1000])

print("\n" + "=" * 50)
print("ПРОВЕРКА TENSORFLOW")
print("=" * 50)

try:
    import tensorflow

    print(f"✅ TensorFlow установлен: {tensorflow.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow НЕ установлен для этого интерпретатора: {e}")

print("\n" + "=" * 50)
print("РЕКОМЕНДАЦИИ")
print("=" * 50)

print("1. Установите TensorFlow через PyCharm:")
print("   File → Settings → Project → Python Interpreter → '+' → ищите 'tensorflow'")
print("\n2. Или через терминал PyCharm:")
print(f"   {sys.executable} -m pip install tensorflow==2.13.0")