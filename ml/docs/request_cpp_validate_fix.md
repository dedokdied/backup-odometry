# Фикс validate_artifact.py

Проверен на моём артефакте: после правки он даёт `ACCEPTED`.

## Причина

`block()` искала первое текстовое вхождение имени:

```python
i = text.find(marker)
j = text.find("{", i)
k = text.find("};", j)
return re.findall(r'"([^"]+)"', text[j:k])
```

Но в `ml_features.hpp` строки 4–7 содержится комментарий:

```cpp
// ... kFeatureNames / kOutputNames arrays below and rejects any model artifact
// ... kFeatureSchemaVersion and updating docs/04_ml_contract.md.
```

Поэтому `find` попадал в комментарий, `{` искался со строки 20, а `};` находился
на строке 46. В блоке 20–46 нет ни одного строкового литерала, поэтому оба массива
приходили пустыми. Счётчики `kNumFeatures` при этом читались верно — там
используется `re.search` с `\s*=\s*(\d+)`, и комментарий их не задевает. Отсюда
странная картина «счётчики верные, имена пустые».

## Правка

```python
def block(marker: str) -> list[str]:
    # Anchor on the declaration, not the first textual occurrence: the header
    # mentions kFeatureNames / kOutputNames in a comment on line 4, so find()
    # lands there and the extracted block is comment text.
    m = re.search(re.escape(marker) + r"\[[^\]]*\]\s*=\s*\{", text)
    if m is None:
        raise SystemExit(
            f"cannot find an array declaration for {marker!r} in {path}: the header "
            f"mentions the name in a comment but never declares it"
        )
    j = m.end() - 1
    k = text.find("};", j)
    if j < 0 or k < 0:
        raise SystemExit(f"cannot find the array after {marker!r}")
    return re.findall(r'"([^"]+)"', text[j:k])
```

Регулярка `\[[^\]]*\]\s*=\s*\{` матчит объявление целиком:
`kFeatureNames[kNumFeatures] = {` — имя, квадратные скобки с любым содержимым
внутри, знак равно с пробелами, открывающая фигурная.

Если будете переносить в свой `D:\HACKATON\tools\validate_artifact.py`, учтите
два момента, на которых я споткнулся при переносе:

- **`re.escape(marker)`** обязателен. Без него строка с именем работает как
  регулярка, и при любом спецсимвере в имени совпадение поедет.
- Удалите старую проверку `if i < 0:` — переменная `i` больше не существует,
  и валидатор упадёт с `NameError` вместо того, чтобы отработать.

## Проверка

```bash
python validate_artifact.py -d models/ml_model.yaml --header cpp/tram_odometry/include/tram_odometry/ml_features.hpp
```

Ожидаемый вывод на моём артефакте:

```
contract (from ml_features.hpp)
  schema_version : 1
  features (16) : ['u', 'v', 'a_model', 'grade', 'mu', 'omega_front', 'omega_rear',
                   'b_scale', 'slip_index', 'trust', 'cmd_rate', 'dt', 'abs_u',
                   'u_sq', 'v_sq', 'force_ratio']
  outputs  (3) : ['a_residual', 'log_scale', 'mu']

descriptor: models\ml_model.yaml
weights    : speed_residual.bin  51 float64  sha256:e035b491a109d8a2
  max |W|  : 10.0551

ACCEPTED: descriptor, weights and contract agree
```

Обратите внимание на `max |W| = 10.0551` — это в 27 раз выше прошлых артефактов.
Это не ошибка валидатора, а предупреждение: возможна опора на выбросы. Подробности
в `request_cpp_load_artifact.md`.
