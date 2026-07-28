# yolo — бриф приложения

Репо `supervisely-ecosystem/yolo` · дефолтная ветка **`master`** (не `main`, как у RT-DETRv2) ·
образ `supervisely/yolo:1.0.33` · версия приложения `2.0.0` · `instance_version` `6.17.8` ·
пин SDK `supervisely==6.74.16`. Снимок на 2026-07-28.

Одной фразой: обучение и инференс моделей YOLO (Ultralytics) — детекция, сегментация, поза.

## Под-приложения

| режим | конфиг | entrypoint | GPU |
|---|---|---|---|
| train | `supervisely_integration/train/config.json` | `python3 -m uvicorn main:train.app --host 0.0.0.0 --port 8000 --ws websockets --app-dir supervisely_integration/train` (`:36`) | `required` (`:24`) |
| serve | `supervisely_integration/serve/config.json` | `TODO: не проверено` | `preferred` (`:24`) |

Оба режима используют один образ (`train/config.json:20`, `serve/config.json:20`).

## Отличия от эталона RT-DETRv2

Общий скелет — `.claude/references/app-anatomy-rtdetr.md`, здесь только расхождения.

| что | как здесь |
|---|---|
| фреймворк обучения | тот же `TrainApp` (`supervisely_integration/train/main.py:26-31`), но обучение делегируется Ultralytics через свой `Trainer` (`train/trainer.py:6-11`) |
| GUI | декларативный из `train/app_options.yaml`, своего кода GUI нет |
| базовый класс инференса | `sly.nn.inference.ObjectDetection` (`serve/serve_yolo.py:17`), у RT-DETRv2 — свой |
| формат данных для трейнера | YOLO, конвертация одной строкой `project.to_yolo(...)` (`train/main.py:74-79`); у RT-DETRv2 — COCO |
| зоо моделей | `supervisely_integration/models.json` — общий для train и serve (`train/main.py:28`, `serve/serve_yolo.py:19`) |
| экспорт | ONNX и TensorRT, оба через хуки `@train.export_onnx` / `@train.export_tensorrt` (`train/main.py:58`, `:66`) |
| дефолтная ветка | `master` |

## Специфика модели — куда смотреть в первую очередь

- `supervisely_integration/train/main.py:74-86` — конвертация в YOLO + подмена путей в настройках Ultralytics (`SettingsManager`)
- `supervisely_integration/train/trainer.py:6-11` — обёртка над `YOLO(...)`, колбэки обучения
- `supervisely_integration/serve/serve_yolo.py:17-21` — `YOLOModel`, откуда берутся модели и настройки инференса
- `supervisely_integration/train/hyperparameters.yaml` — гиперпараметры
- `supervisely_integration/train/app_options.yaml` — опции GUI (`default_model: YOLO26s-det`)
- `supervisely_integration/train/yolo_settings.json` — настройки Ultralytics, переписываются на старте

## Запуск и отладка

```bash
cd apps/yolo
# TODO: не проверено — есть ли create_venv.sh; ставить из dev_requirements.txt
python3 -m uvicorn main:train.app --host 0.0.0.0 --port 8000 --ws websockets --app-dir supervisely_integration/train
```

Окружение: `local.env` в репо + `~/supervisely.env`. Отладка против локального SDK и ловушки
симлинков — `.claude/references/dev-loop.md`.

## Грабли этого приложения

- `docker/Dockerfile.deploy:1` собирается `FROM supervisely/yolo:1.0.31` — отстал от текущего
  `1.0.33` в `publish.sh` и конфигах.
- `dev_requirements.txt:1` держит закомментированный git-пин SDK на тестовую ветку — след прошлых
  отладок, легко случайно раскомментировать.
- Дефолтная ветка `master`: ветвиться и открывать PR надо от неё, автоматика релиза ветки
  срабатывает на push любой ветки кроме `main`/`master` (`.github/workflows/release_branch.yml`).

## Релиз

Две дорожки, как у эталона: образ — `docker/publish.sh` (тег правится в обеих строках, пуш делает
пользователь), приложение — релиз ветки/тега через `.github/workflows/release_branch.yml` и
`release.yml`. Порядок выкатки и связь версий — `.claude/references/release-and-versions.md`.
