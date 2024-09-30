# Запуск обучения

## Обучение для задачи общей классификации:.


**Запустите скрипт с указанием путей к данным и гиперпараметров**:

   Пример команды для запуска:
   ```bash
   python tools/train_classification.py --train_dataset1 /path/to/dataset1/train \
                    --train_dataset2 /path/to/dataset2/train \
                    --train_dataset3 /path/to/dataset3/train \
                    --val_dataset1 /path/to/dataset1/val \
                    --val_dataset2 /path/to/dataset2/val \
                    --val_dataset3 /path/to/dataset3/val \
                    --output_dir /path/to/output \
                    --pretrain_ratio 0.8 \
                    --pretrain_lr 0.02 \
                    --pretrain_mask_type entmax \
                    --pretrain_verbose 10 \
                    --n_d 64 \
                    --n_a 64 \
                    --n_steps 5 \
                    --gamma 1.5 \
                    --lambda_sparse 1e-4 \
                    --lr 0.02 \
                    --step_size 10 \
                    --gamma_lr 0.9 \
                    --patience 30 \
                    --batch_size 128 \
                    --virtual_batch_size 256 \
                    --verbose 10
   ```

#### Описание аргументов:

- **`--train_dataset1`**: Путь к тренировочным данным первого датасета (лейбл 0).
- **`--train_dataset2`**: Путь к тренировочным данным второго датасета (лейбл 1).
- **`--train_dataset3`**: Путь к тренировочным данным третьего датасета (лейбл 2).

- **`--val_dataset1`**: Путь к валидационным данным первого датасета.
- **`--val_dataset2`**: Путь к валидационным данным второго датасета.
- **`--val_dataset3`**: Путь к валидационным данным третьего датасета.

- **`--output_dir`**: Путь, где будут сохранены обученная модель и другие результаты.

##### Параметры предобучения:

- **`--pretrain_ratio`**: Соотношение данных для предобучения (по умолчанию 0.8, что означает 80% данных для предобучения, 20% для валидации).
- **`--pretrain_lr`**: Скорость обучения для этапа предобучения (по умолчанию 0.02).
- **`--pretrain_mask_type`**: Тип маски для предобучения (`entmax` или `sparsemax`, по умолчанию `entmax`).
- **`--pretrain_verbose`**: Уровень детализации предобучения (по умолчанию 10).

##### Параметры для TabNet:

- **`--n_d`**: Количество признаков для слоев принятия решений (по умолчанию 64).
- **`--n_a`**: Количество признаков для слоев внимания (по умолчанию 64).
- **`--n_steps`**: Количество шагов принятия решений (по умолчанию 5).
- **`--gamma`**: Параметр gamma для контроля влияния прошлых шагов на текущий (по умолчанию 1.5).
- **`--lambda_sparse`**: Параметр регуляризации для контроля разреженности (по умолчанию 1e-4).

##### Параметры обучения:

- **`--lr`**: Скорость обучения (по умолчанию 0.02).
- **`--step_size`**: Количество шагов для изменения скорости обучения (по умолчанию 10).
- **`--gamma_lr`**: Коэффициент уменьшения скорости обучения (по умолчанию 0.9).
- **`--patience`**: Количество эпох без улучшений для остановки обучения (по умолчанию 30).

##### Batch Size:

- **`--batch_size`**: Размер батча для обучения (по умолчанию 128).
- **`--virtual_batch_size`**: Виртуальный размер батча для обработки больших данных (по умолчанию 256).

##### Прочие параметры:

- **`--verbose`**: Уровень детализации вывода в процессе обучения (по умолчанию 10).

---

### Результаты:
- Обученная модель будет сохранена в указанную директорию `output_dir` под названием `classification_model.zip`.
- Модель будет использовать три класса: 0, 1 и 2, которые соответствуют трем датасетам, указанным в аргументах `--train_dataset1`, `--train_dataset2`, и `--train_dataset3`.

### Пример полного вызова:

```bash
python tools/train_classification.py --train_dataset1 /data/dataset1/train \
                 --train_dataset2 /data/dataset2/train \
                 --train_dataset3 /data/dataset3/train \
                 --val_dataset1 /data/dataset1/val \
                 --val_dataset2 /data/dataset2/val \
                 --val_dataset3 /data/dataset3/val \
                 --output_dir /models/output \
                 --pretrain_ratio 0.8 --pretrain_lr 0.02 --pretrain_mask_type entmax --pretrain_verbose 10 \
                 --n_d 64 --n_a 64 --n_steps 5 --gamma 1.5 --lambda_sparse 1e-4 \
                 --lr 0.02 --step_size 10 --gamma_lr 0.9 --patience 30 \
                 --batch_size 128 --virtual_batch_size 256 --verbose 10
```
