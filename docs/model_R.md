# Модель R: справка

Модель R обучена вне этого репозитория, в исходном проекте. Здесь записано, как она получена,
чтобы было понятно, что именно оценивается. Код обучения и корпуса в репозиторий не входят,
поэтому воспроизвести обучение по этому репозиторию нельзя.

| | |
|---|---|
| база | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| метод | LoRA r = 128, α = 256, dropout 0.05, набор целевых слоёв `wide`; вес потерь предсказателя остаточных кодбуков 0.3 |
| данные | 18 013 клипов (ESD, CREMA-D, RAVDESS), состав в `docs/train_manifest_stats.txt` |
| оптимизация | lr 5·10⁻⁵ без расписания, батч 2 с накоплением 16, 3 эпохи, weight decay 0.01, сид 42 |
| канал управления | инструкция с одной эмоцией на клип; в манифесте три формы строки: фраза (7 289), тег (7 276), без инструкции (3 448) |
| чекпоинт | шаг 1050 (вторая эпоха), после слияния адаптера с базой; sha256 файла весов `475e2ef4a766fc24622045aa52bdffd879de9866af1b763b607925d427b01e49` |
| диктор при синтезе | `esd0016` |
| где лежит | [byebye9/qwen3-tts-emotion-instruct](https://huggingface.co/byebye9/qwen3-tts-emotion-instruct), ревизия `47f851a9c8f32fb8a720c479e5a3509aa18cbd4b` |

Команда обучения из журнала исходного проекта (пути относятся к тому репозиторию):

```bash
python scripts/train_lora_official.py \
  --init_model_path models/Qwen3-TTS-12Hz-0.6B-Base \
  --train_jsonl data/t1_official_en10rc_instruct_tagadj/train_with_codes.jsonl \
  --val_jsonl data/t1_official_en10rc_instruct_tagadj/val_with_codes.jsonl --val_limit 100 \
  --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 --lora_targets wide --lr_schedule constant \
  --batch_size 2 --grad_accum 16 --lr 5e-5 --num_epochs 3 --warmup_steps 0 --weight_decay 0.01 \
  --sub_loss_weight 0.3 --loss_variant fixed --save_every_steps 150 --seed 42 --save_mode adapter \
  --speaker_name esd0016 --instruct
```

Шаг 1050 выбран в исходном проекте по точности распознавания категории эмоции на отдельном наборе
из пяти текстов (0.74 при шансе 0.20) и по её устойчивости на других текстах и сидах. Смена эмоции
внутри высказывания при выборе не измерялась.
