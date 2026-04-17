from __future__ import annotations


APP_DISPLAY_NAME = "Минимальная канбан-доска"
APP_SUBTITLE = "Локальная доска задач с обратным отсчётом и встроенным JSON API"
API_LABEL_PREFIX = "Локальный API"

ERROR_TITLE = "Ошибка"
ERROR_VALIDATION_TITLE = "Проверьте данные"
STARTUP_ERROR_TITLE = "Ошибка запуска"
STARTUP_ERROR_MESSAGE = (
    "Не удалось запустить локальный API. Закройте другие копии программы и попробуйте снова."
)
UNEXPECTED_ERROR_TITLE = "Непредвиденная ошибка"
UNEXPECTED_ERROR_MESSAGE = (
    "Произошла непредвиденная ошибка. Подробности сохранены в журнале программы."
)
BOARD_LOAD_ERROR_MESSAGE = (
    "Не удалось прочитать состояние доски. Подробности сохранены в журнале программы."
)

# Override the legacy desktop title while keeping the existing data paths unchanged.
APP_DISPLAY_NAME = "AutoStop CRM"

COLUMN_LABELS_RU = {
    "inbox": "Входящие",
    "in_progress": "В работе",
    "control": "На контроле",
    "done": "Готово",
}

DEFAULT_COLUMN_EMPTY_MESSAGES = {
    "inbox": "Здесь пока нет карточек.\nСоздайте первую задачу или добавьте её через API.",
    "in_progress": "Здесь пока нет задач в работе.\nПеретащите сюда карточку из другого столбца.",
    "control": "Контрольных задач пока нет.\nСюда удобно переносить карточки, по которым нужно вернуться позже.",
    "done": "Завершённых карточек пока нет.\nПеретащите сюда готовую задачу.",
}


def get_column_empty_message(column_id: str, label: str) -> str:
    if column_id in DEFAULT_COLUMN_EMPTY_MESSAGES:
        return DEFAULT_COLUMN_EMPTY_MESSAGES[column_id]
    return f"В столбце «{label}» пока нет карточек.\nПеретащите сюда задачу или создайте новую."


STATUS_LABELS_RU = {
    "ok": "Времени достаточно",
    "warning": "Срок подходит к концу",
    "expired": "Срок истёк",
}

BUTTON_HELP = "Помощь"
BUTTON_NEW_CARD = "Новая карточка"
BUTTON_NEW_COLUMN = "Новый столбец"
BUTTON_ARCHIVE = "В архив"
BUTTON_BACK = "Назад"
BUTTON_NEXT = "Далее"
BUTTON_FINISH = "Готово"
BUTTON_SAVE = "Сохранить"
BUTTON_CANCEL = "Отмена"

TOOLTIP_HELP = "Открыть встроенную инструкцию"
TOOLTIP_NEW_CARD = "Создать новую карточку"
TOOLTIP_NEW_COLUMN = "Добавить новый столбец на доску"
TOOLTIP_DRAG_CARD = "Перетащите карточку в другой столбец или дважды щёлкните для редактирования"
TOOLTIP_ARCHIVE = "Переместить карточку в архив"

CARD_DIALOG_CREATE_TITLE = "Новая карточка"
CARD_DIALOG_EDIT_TITLE = "Карточка"
COLUMN_DIALOG_TITLE = "Новый столбец"
CARD_FIELD_TITLE = "Заголовок"
CARD_FIELD_DESCRIPTION = "Описание"
CARD_FIELD_DEADLINE = "Срок"
CARD_FIELD_DEADLINE_DAYS = "Дни"
CARD_FIELD_DEADLINE_HOURS = "Часы"
COLUMN_FIELD_LABEL = "Название столбца"
CARD_PLACEHOLDER_TITLE = "Введите заголовок карточки"
CARD_PLACEHOLDER_DESCRIPTION = "Введите описание карточки"
COLUMN_PLACEHOLDER_LABEL = "Введите название нового столбца"
CARD_NO_DESCRIPTION = "Описание не указано"
CARD_STATUS_TOOLTIP_TEMPLATE = "Состояние: {label}"

CARD_TITLE_EMPTY_MESSAGE = "Заголовок карточки не должен быть пустым."
CARD_TITLE_LONG_MESSAGE = "Заголовок карточки не должен превышать 120 символов."
CARD_DESCRIPTION_LONG_MESSAGE = "Описание карточки не должно превышать 5000 символов."
CARD_DEADLINE_INVALID_MESSAGE = "Укажите срок больше нуля."
COLUMN_LABEL_EMPTY_MESSAGE = "Название столбца не должно быть пустым."
COLUMN_LABEL_LONG_MESSAGE = "Название столбца не должно превышать 40 символов."

HELP_DIALOG_TITLE = "Быстрый старт"
HELP_PROGRESS_TEMPLATE = "Шаг {current} из {total}"

ONBOARDING_PAGES = [
    (
        "Что это за программа",
        "Это локальная минималистичная канбан-доска для быстрых личных задач. "
        "Программа работает полностью на вашем компьютере и не требует ручного запуска сервера.",
    ),
    (
        "Как создать карточку",
        "Нажмите кнопку «Новая карточка» в правом верхнем углу. "
        "Введите заголовок, при необходимости добавьте описание и задайте срок в днях и часах.",
    ),
    (
        "Как работает отсчёт",
        "После сохранения карточки обратный отсчёт начинается автоматически. "
        "В карточке всегда видно, сколько времени осталось до срока.",
    ),
    (
        "Как читать лампочку состояния",
        "Лампочка меняется автоматически: зелёная означает, что времени достаточно, "
        "жёлтая показывает приближение срока, а красная означает просрочку.",
    ),
    (
        "Как работать со столбцами",
        "На доске можно создавать новые столбцы кнопкой «Новый столбец». "
        "Карточки можно перетаскивать мышью в любой существующий столбец.",
    ),
    (
        "Как редактировать и архивировать карточку",
        "Дважды щёлкните по карточке, чтобы открыть её. "
        "В окне карточки можно изменить заголовок, описание и срок, а также отправить карточку в архив.",
    ),
    (
        "Готовность к интеграции с ChatGPT",
        "Вместе с интерфейсом программа поднимает локальный JSON API по адресу {api_url}. "
        "Бизнес-логика и хранение отделены от UI, поэтому приложение уже подготовлено к будущему function calling.",
    ),
]

BUTTON_APPLY = "Применить"
BUTTON_RESET_DEFAULTS = "Сбросить по умолчанию"
BUTTON_TEST_CONNECTION = "Проверить соединение"
BUTTON_SHOW_SECRET = "Показать"
BUTTON_HIDE_SECRET = "Скрыть"
BUTTON_COPY = "Копировать"
BUTTON_SETTINGS = "Настройки"

TOOLTIP_SETTINGS = "Открыть настройки интеграции с ChatGPT, OpenAI и MCP"

SETTINGS_WINDOW_TITLE = "Настройки интеграции"
SETTINGS_WINDOW_SUBTITLE = "Параметры подключения к локальному API, MCP и OpenAI-совместимому API"
SETTINGS_SECTION_GENERAL = "Общие"
SETTINGS_SECTION_CREDENTIALS = "Доступ и токены"
SETTINGS_SECTION_OPENAI = "ChatGPT / OpenAI"
SETTINGS_SECTION_MCP = "MCP"
SETTINGS_SECTION_DIAGNOSTICS = "Диагностика"

SETTINGS_FIELD_INTEGRATION_ENABLED = "Включить интеграцию"
SETTINGS_FIELD_USE_LOCAL_API = "Использовать локальный API"
SETTINGS_FIELD_LOCAL_API_HOST = "Хост локального API"
SETTINGS_FIELD_LOCAL_API_PORT = "Порт локального API"
SETTINGS_FIELD_LOCAL_API_URL = "Адрес локального API"
SETTINGS_FIELD_AUTO_CONNECT = "Автоподключение при запуске"
SETTINGS_FIELD_TEST_MODE = "Тестовый режим"

SETTINGS_FIELD_OPENAI_API_KEY = "Ключ API OpenAI"
SETTINGS_FIELD_LOCAL_API_TOKEN = "Токен Bearer локального API"
SETTINGS_FIELD_MCP_TOKEN = "Токен Bearer MCP"

SETTINGS_FIELD_PROVIDER = "Провайдер"
SETTINGS_FIELD_MODEL = "Модель"
SETTINGS_FIELD_BASE_URL = "Базовый URL"
SETTINGS_FIELD_ORGANIZATION = "Organization ID"
SETTINGS_FIELD_PROJECT = "Project ID"
SETTINGS_FIELD_TIMEOUT = "Timeout, сек"

SETTINGS_FIELD_MCP_ENABLED = "Включить MCP"
SETTINGS_FIELD_MCP_HOST = "Хост MCP"
SETTINGS_FIELD_MCP_PORT = "Порт MCP"
SETTINGS_FIELD_MCP_PATH = "Путь MCP"
SETTINGS_FIELD_MCP_ENDPOINT = "Локальный адрес MCP"
SETTINGS_FIELD_MCP_PUBLIC_URL = "Публичный HTTPS-адрес или адрес туннеля"

SETTINGS_FIELD_SETTINGS_FILE = "Файл настроек"
SETTINGS_FIELD_RUNTIME_API = "Текущий адрес API приложения"
SETTINGS_FIELD_LAST_TEST_AT = "Последняя проверка"
SETTINGS_FIELD_LOCAL_API_STATUS = "Статус локального API"
SETTINGS_FIELD_MCP_STATUS = "Статус MCP"
SETTINGS_FIELD_OPENAI_STATUS = "Статус OpenAI"

SETTINGS_STATUS_NOT_TESTED = "Не проверялось"
SETTINGS_STATUS_SUCCESS = "Успешно"
SETTINGS_STATUS_FAILED = "Ошибка"
SETTINGS_STATUS_SKIPPED = "Пропущено"

SETTINGS_VALIDATION_TITLE = "Ошибки в настройках"
SETTINGS_VALIDATION_PREFIX = "Исправьте следующие поля:"
SETTINGS_RESET_TITLE = "Сброс настроек"
SETTINGS_RESET_MESSAGE = "Вернуть настройки интеграции к значениям по умолчанию? Несохранённые изменения будут потеряны."
SETTINGS_TEST_TITLE = "Результат проверки соединения"
SETTINGS_SAVE_SUCCESS = "Настройки сохранены."
SETTINGS_APPLY_SUCCESS = "Настройки применены."
SETTINGS_RESET_READY = "В форму подставлены значения по умолчанию. Нажмите «Применить» или «Сохранить»."
SETTINGS_TEST_SUCCESS = "Проверка соединения завершена."
SETTINGS_COPY_READY = "Значение скопировано в буфер обмена."

SETTINGS_RUNTIME_NOTE = (
    "Изменения host, port и токенов применяются к следующим запускам и внешним подключениям. "
    "Уже запущенный локальный API текущего окна продолжает работать с текущими параметрами до перезапуска программы."
)
SETTINGS_SECRET_NOTE = (
    "Секреты пока сохраняются в settings.json без системного шифрования. "
    "Архитектура подготовлена к будущей интеграции с Windows Credential Manager, DPAPI или OAuth."
)

SETTINGS_CONNECTION_SUMMARY_TEMPLATE = (
    "Локальный API: {local_api}\n"
    "MCP: {mcp}\n"
    "OpenAI: {openai}"
)
