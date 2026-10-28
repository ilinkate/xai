# xai
В папке находятся два самостоятельных скрипта:

- `build_dataset.py` создаёт полный датасет и выборки train, validation, test.
- `train_model.py` обучает модели, оценивает результат и строит графики SHAP/LIME.

Прежние скрипты и папки с результатами для работы этого комплекта не нужны. Они сохранены в родительской папке как история эксперимента. Новый код не импортирует и не запускает старые скрипты.

## Быстрый запуск

Откройте терминал и перейдите в рабочую папку.
Для успешного запуска в рабочей папке должен быть следующий список файлов и папок:
```bash
build_dataset.py
train_model.py
source-|
       |-source_manifest.json
       |-wordpress_records.jsonl
```

```bash
cd "/././."
python3 build_dataset.py
python3 train_model.py
```

Используется Python 3.12. Проверенные версии библиотек перечислены в `requirements.txt`. Если нужно отдельное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_dataset.py
python train_model.py
```
