# Video Classification with TabNet and CLIP

Overview

Этот проект реализует систему классификации видео с использованием модели TabNet и модели для извлечения эмбеддингов CLIP. Видео конвертируются в изображения, из которых извлекаются эмбеддинги с помощью CLIP, после чего классификация осуществляется с помощью TabNet.

Features

	•	Извлечение эмбеддингов из изображений с использованием модели CLIP.
	•	Предобучение и обучение модели TabNet.
	•	Поддержка обучения по любому выбранному лейблу, который извлекается из структуры директорий.
	•	Поддержка сохранения и загрузки LabelEncoder для преобразования строковых лейблов в числовые.
	•	Возможность предобучения модели с дальнейшим обучением для более точной классификации.

Project Structure

project_root/
├── src/
│   ├── __init__.py                # Инициализация пакета
│   ├── model.py                   # Класс для модели TabNet и методов инференса
│   ├── preprocess.py              # Функции для предобработки изображений и извлечения эмбеддингов
│   ├── train_model.py             # Основная логика обучения модели
│   └── utils/
│       ├── __init__.py            # Инициализация для утилит
│       ├── preprocess.py          # Препроцессинг данных
│       ├── crop_frames.py         # Нарезка видео на кадры
│       └── split_videos.py        # Разбить папку с данными на train/val/test
├── examples/
│   └── inference.ipynb            # Пример использования модели для инференса
├── main.py                        # Точка входа для обучения модели
├── requirements.txt               # Зависимости проекта
└── README.md                      # Этот файл

## Installation

1. Клонирование репозитория

```
git clone hhttps://github.com/ShockOfWave/bubbles_champagne.git
cd bubbles_champagne
```

2. Установка зависимостей

Перед тем, как начать, убедитесь, что у вас установлены все необходимые зависимости. Для этого выполните:

```
pip install -r requirements.txt
```

3. Проверка установки

Убедитесь, что все зависимости установлены корректно, и проект готов к использованию. В этом проекте используется Python версии 3.8 или выше.

## Usage

1. Запуск обучения

Запустите скрипт main.py для обучения модели. Убедитесь, что вы передали корректные пути к данным и директорию для сохранения модели:

```
python main.py --root_dir /path/to/data \
               --checkpoints /path/to/checkpoints \
```

Параметры:

	•	--root_dir: Путь к данным.
	•	--checkpoints: Путь для сохранения модели.

3. Инференс

Для инференса данных используйте inference.ipynb, чтобы загрузить сохраненную модель и провести классификацию на новых изображениях.


License

Этот проект лицензирован под MIT License.

Этот файл README.md включает информацию о структуре проекта, установке, использовании, а также общие проблемы, с которыми вы можете столкнуться при работе с проектом.