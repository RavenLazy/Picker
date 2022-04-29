#!/usr/local/bin/python3.10
#  coding: utf-8

# Автор: Макарцев Павел
# lazy2005.lazys@gmail.com

"""Программа для поиска и переноса во временную папку старых файлов.

Управление осуществляется командной строкой и специальными файлами .rule (имя можно изменить) расположенными в
папках поиска. Если дата файла или каталога старше чем заданная в .rule, то объект удаляется или переносится
во временную папку
Файл .rule текстовый файл содержащий:
    (Prefix)Folder:Time:IS_Lock:level
        Prefix::

            '-' : Пропускать файл или каталог
            '+' : Проверять файл или каталог
            '!' : Проверять по самому "новому" файлу
            '@' : Проверять только дату каталога

        Time::

            Формат - число: если разница между текущей датой и датой модификации (по умолчанию) больше этого числа -
            файл считается устаревшим. За числом код времени.
                H - час (60 сек)
                D - день (H * 24)
                M - месяц (D * 30)
                Y - год (D * 365) или (M * 12)

        IS_Lock::

            L : Не удалять каталог, если пустой. По умолчанию
            U : Можно удалять

        level::

            На какую глубину действует это правило с сохранением входных данных. Иначе работает правило для всех (*)
                -1 - На всю глубину
                 0 - Только текущий уровень

    Пример::

        -Test:1H        Пропускать каталог пока он младше 1 часа. Каталог не удалять
        +Test:2D        Проверять каталог если он старше 2 дней. Не удалять
        !Test1:1M:U     Искать самый молодой файл, если он будет старше 1 месяца - удалить все файлы. Каталог,
                        если пустой, удалить.
        @Test2:1Y:L     Если каталог старше 1 года, сразу удалить (если это временная папка) или перенести. Файлы
                        внутри не проверяем.
        +Test3:2H::1    Проверять каталог если старше 2 часов, это же правило перенести внутрь каталога на 1 уровень.
                        То есть внутри объекты будут проверяться по этому же правилу. Далее правило будет +*:2H:U:-1.
        +Test4:1D:U:-1  Все файлы и каталоги будут проверяться по правилу +*:1D:U:-1

    Формат командной строки:

        picker test | execute start_folder parameters
            - start_folder : абсолютный путь. Откуда начинаем искать

            - test : тестирование, без переноса или удаления. Создается временная папка
            - execute: выполнение

            Parameters::

                -Search : папки поиска (относительно start_folder). Можно использовать совместно с
                -pathname : файл, аналог, Search. Лежит в start_folder
                -rule : переименовать файл .rule
                -trash : переименовать временную папку. Создается в start_folder
                -console : выводим информацию в консоль
                -log : создаем лог-файл
                -overall : Вывести только общий результат: <Выключено>
                -bot : Отправка результата в телеграмм-канал: <Выключено>
                -zero : Вывод информации и с нулевыми значениями: <Выключено>
                -name : Имя лог-файла: <log.log>
                -size : Размер лога: <10000>
                -append : Продолжать дописывать: <Включено>
"""


import math
import subprocess
from sys import platform, exit, argv
import re
import datetime
from pathlib import Path
import argparse
from shutil import rmtree
import traceback

VERSION = '1.0d'
PROG = Path(__file__).name

CREATION_TIME = 0
MODIFICATION_TIME = 1   # Использовать по умолчанию
ACCESS_TIME = 2

NAME_MOVE_DIR = "Trash"
DAY_TO_PURGE = 14
Default_rule_old_days = f"@*:1D:U\n-*:{DAY_TO_PURGE}D:U"

NAME_RULE = ".rule"
NAME_PATH = "Software/.path"
ERROR_LOG = Path(PROG).stem + '_error.log'
NAME_LOG = Path(PROG).stem + '.log'

CURRENT_DATE = datetime.datetime.now().replace(second=0, microsecond=0).timestamp()


MAIN_EXCLUDE = [NAME_RULE, NAME_PATH, PROG]

list_includes_znak = ("@", "!", "+")

old_date_pattern = re.compile(r"\d{2}-\d{2}-\d{4}$")

width_text = 200

mask = {'log':      0b001,
        'console':  0b010,
        'bot':      0b100}

int_log = 1
int_console = 2
int_bot = 3


def add_default_rule():
    if not arguments.trash.exists():
        arguments.trash.mkdir(parents=True)
    if not (rule := arguments.trash.joinpath(arguments.rule)).exists():
        with rule.open("w") as f:
            f.writelines(Default_rule_old_days)
    return None


def decompress(el: list, folders: Path, dp: bool):
    # Раскладываем время из файла правил
    name: str
    count = 1
    for item in el:
        name, dates, lock, deep = (a := re.split(r":", item, maxsplit=3)) + [''] * (4 - len(a))
        deep = int(deep or '0')
        dates = dates or '0N'
        if dp is False:
            try:
                dates = int(normal_date(dates)) or (0 if name[0] in list_includes_znak else int(CURRENT_DATE))
            except ValueError:
                logger.write_log(f"{item!r} ошибка разметки в списке правил {NAME_RULE!r}."
                                 f"\nПапка: {folders.as_posix()!r}.\nСтрока: {count}")
                logger.send_all()
                exit(1)
            lock = False if lock == "U" else True  # True - папка не будет удалена
        count += 1
        yield name, (dates, lock, deep)


def return_list(text, folders, deep=False):
    tmp = []
    for name, args in decompress(text, folders, deep):
        if name in tmp:
            continue
        tmp.append(name)
        yield name, args


def for_size(item):
    # Переводим буквенный код размера файла в числовой
    ident: str
    size: int
    coefficient = 1000
    size, ident = re.findall(r"(\d+)([MmKkBb]?)", item)[0]
    size = int(size)
    return {"m": size * (coefficient ** 2), "k": size * coefficient, "b": size * 1}.get(ident.lower(), size)


def create_path(namespace, values):
    line = []
    if values:
        folder = getattr(namespace, 'folder')
        for v in values:
            line.append(folder.joinpath(v))
    return line


def read_path(namespace, values):
    line = []
    for files in create_path(namespace, [values]):
        if files.exists():
            with open(files, 'r', encoding='utf-8') as f:
                line.extend(create_path(namespace, f.read().splitlines()))
    return line


def set_bit(num, pos):
    return num | 1 << pos


def re_change(args: argparse.ArgumentParser, line, value):
    ret = args.parse_args([line.command, line.folder.as_posix(), *value])
    return ret


def alternative_parsers():
    prog_name = PROG.split('.py')[0]
    default = {True: 'Включено', False: 'Выключено'}

    arg = argparse.ArgumentParser(description=f"Перенос старых файлов в отдельную директорию с сохранением путей. "
                                              f"%(prog)s версия {VERSION}", prog=prog_name)

    search_folder = arg.add_argument_group('Main command')
    logged = arg.add_argument_group('Logger', 'Управление выводом в лог.')

    arg.add_argument("command", choices=["execute", "test"],
                     help="<execute> работа по заданным правилам. <test> прогон по папкам без изменения.")

    arg.add_argument("folder", metavar="<start folder name>", action="store",
                     type=Path, help=f"Откуда начинаем искать")

    search_folder.add_argument("-Search", type=str, action=ActionSearch,
                               metavar="<Folder1>...<FolderNNN>", help=f"Папки поиска.")

    search_folder.add_argument("-pathname", type=str, default='.path', metavar=".path", action=ActionFile,
                               help=f"Имя файла с папками для поиска: <%(default)s>")

    arg.add_argument("-rule", default='.rule', metavar="<'.rule'>",
                     help="Имя файла с правилами: %(metavar)s")

    arg.add_argument("-trash", nargs=1, metavar="<'Trash'>", type=str, action=ActionTrash,
                     help=f"Имя папки в которую будут переносится старые файлы: %(metavar)s")

    logged.add_argument("-console", nargs='*', default=0, type=int, action=ActionCount, const=[2],
                        help=f"Выводить в консоль: -c = %(const)s, -c 1 = размеры больше нуля, "
                             f"-c 2 = не нулевые размеры, -c 3 = выводить всё, [4] - выводить информацию.")

    logged.add_argument("-log", action=ActionCount, default=0, const=[2], type=int, nargs='*',
                        help=f"Писать в файл: -l = %(const)s, -l 1 = размеры больше нуля, "
                             f"-l 2 = не нулевые размеры, -l 3 = выводить всё, [4] - выводить информацию.")

    logged.add_argument("-bot", action=ActionCount, default=0, nargs='*', const=[1], type=int,
                        help=f"Отправка результата в телеграмм-канал: -b = %(const)s, -b 1 = размеры больше нуля, "
                             f"-b 2 = не нулевые размеры, -b 3 = выводить всё, [4] - выводить информацию.")

    arg.add_argument('--version', "-V", action='version', version=f'%(prog)s {VERSION}')

    # Выставляем параметры для лог файла, если указан аргумент -log

    logged.add_argument("-name", nargs=1, action=ActionTrash, help=f"Имя лог-файла: <{NAME_LOG}>")
    logged.add_argument("-size", default=10000, type=for_size, help="Размер лога: <%(default)s>")
    logged.add_argument("-append", action="store_false", help=f"Выключить дозапись в лог: <{default[True]}>")

    command_argument = arg.parse_args()
    # Имя лог файла
    if command_argument.log and command_argument.name is None:
        arg.set_defaults(name=re_change(arg, command_argument, ['-n', NAME_LOG]).name)

    # Добавляем имя файла с ошибками
    arg.set_defaults(error_log=command_argument.folder.joinpath(ERROR_LOG))

    # Создаем список папок для поиска. Читаем файл .path или что задано в pathname

    arg.set_defaults(Search=re_change(arg, command_argument, ['-p', '.path']).Search)

    if command_argument.trash is None:
        arg.set_defaults(trash=re_change(arg, command_argument, ['-t', NAME_MOVE_DIR]).trash)

    arg.set_defaults(trash_day=(arg.get_default('trash') or command_argument.trash).joinpath(STR_NOW_DATE))
    s = (arg.get_default('Search') or command_argument.Search) + [(arg.get_default('trash') or command_argument.trash)]
    arg.set_defaults(Search=s)
    arg.set_defaults(work=command_argument.command == 'execute')

    return arg.parse_args()


def create_log():
    if arguments.append:
        if arguments.log and arguments.name.exists():
            if arguments.name.stat().st_size >= arguments.size:
                name = arguments.name.as_posix().split(".")[0]
                fullname = Path('-'.join([name, STR_NOW_DATE + arguments.name.suffix]))
                count = 0
                while fullname.exists():
                    count += 1
                    fullname = Path(
                        '-'.join([name, STR_NOW_DATE + f"_{str(count).zfill(3)}{arguments.name.suffix}"]))
                arguments.name.replace_files(fullname)
    elif arguments.name and arguments.name.exists():
        arguments.name.unlink(missing_ok=True)


def get_bit(num: int, pos: int) -> int:
    return num >> pos & 1


def save_log(text, name):
    with open(name, encoding='utf-8', mode='a') as f:
        f.writelines(text)


def send_message(message):
    if platform != "win32" and message:
        path = Path(argv[0]).parent.joinpath('picker_bot.sh')
        txt = map(lambda x: re.sub(r"[-+*#=]*", "", x), message)
        txt = ''.join(map(lambda x: re.sub(r" {2,}", " ", x), txt))
        subprocess.call([path, txt])


def return_time_file(name: Path, type_time):
    return {ACCESS_TIME: name.stat().st_atime, MODIFICATION_TIME: name.stat().st_mtime,
            CREATION_TIME: name.stat().st_ctime}[type_time]


def replace_template(pat, item):
    # Символы в item меняем на подстановочные знаки в pat
    for elem in item:
        for key, value in pat.items():
            elem = elem.replace(key, value)
        yield elem


def normal_date(item: str):
    # Переводим заданное время в секунды
    codes: str
    times: str | float

    times, codes = re.findall(r"([-\d]+)+(\w)", item)[0]

    if codes.lower() == "n":
        return CURRENT_DATE

    return float(times) * {"h": 3600,
                           "d": 86400,
                           "m": 2592000,
                           "y": 31557600}.get(codes.lower(), 0)


def get_count(elem: Path):
    ret = Counter()
    if elem.is_dir():
        ret.total = len(obj := [(files, files.is_file()) for files in elem.iterdir()])
        ret.files = len([x for x in obj if x[1]])
        ret.folder = ret.total - ret.files
    else:
        ret.total = 1
        ret.files = 1
    return ret


def human_read_format(size):
    suff = ["Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ", "ЭБ", "ЗБ", "ЙБ"]
    if size == 0:
        return f"0 {suff[0]}"
    pwr = math.floor(math.log(size, 1024))
    if size > 1024 ** (len(suff) - 1):
        return "не знаю как назвать такое число :)"
    return f"{size / 1024 ** pwr:.2f} {suff[pwr]}"


def change_parent_equal(item: dict) -> dict:
    ret = {f"{key}*": (value[0], value[1] if value[2] > 0 else False, value[2]) for key, value in item.items()}
    return ret


def delta(znak, elem, max_time=None):
    if isinstance(elem, dict):
        fl = elem.get(znak, False)
    else:
        fl = elem.equals.get(znak, False)
    if fl is False:
        return False
    date, *_ = fl
    if max_time:
        return int(CURRENT_DATE - max_time) > date
    ret = int(CURRENT_DATE - return_time_file(elem.folders, MODIFICATION_TIME))
    return ret <= date if znak == '-' else ret > date


def _add_znak(item, znak):
    return list(map(lambda x: znak + x if x[0] not in list_includes_znak else x, item))


def get_deep(rule_text, dp):
    tmp = []
    rule_text += [elem for elem in dp]
    if len(rule_text) == 0:
        yield '', ''
    for name, args in decompress(rule_text, arguments.folder, True):
        date, lock, deep = args

        if name in tmp:
            continue
        tmp.append(name)
        lock = 'L' if len(lock) == 0 else lock
        if deep - 1 != -1:
            if deep > 0:
                deep -= 1
            rl = ":".join([name, date, lock, str(deep)])
            yield rl, rl
        else:
            yield ":".join([name, date, lock, '0']), ''


def reduce(elem: Path, files, folders):
    a = {True: (1, 0), False: (0, 1)}[elem.is_file()]
    return map(sum, zip(a, (files, folders)))


class ActionFile(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        values = read_path(namespace, values)
        atr = create_path(namespace, getattr(namespace, 'Search')) or values
        if len(set(values).intersection(atr)) == 0:
            atr += values
        setattr(namespace, 'Search', atr)


class ActionSearch(ActionFile):

    def __call__(self, parser, namespace, values: list, option_string=None):
        values = create_path(namespace, [values])
        atr = create_path(namespace, getattr(namespace, 'Search')) or values
        if len(set(values).intersection(atr)) == 0:
            atr += values
        setattr(namespace, 'Search', atr)


class ActionTrash(ActionFile):

    def __call__(self, parser, namespace, values, option_string=None):
        values = list(create_path(namespace, values))[0]
        setattr(namespace, self.dest, values)


class ActionCount(ActionFile):

    def __call__(self, parser, namespace, values, option_string=None):
        four = 4 in values
        values = max(set(values).intersection([1, 2, 3]) or self.const)
        bit = (0, 0b10, 0b110, 0b1110)[values]
        values = (bit | 0b10000 if four else bit) | getattr(namespace, self.dest)
        setattr(namespace, self.dest, values)


class Logger:
    __slots__ = ("text", "bot", "log", "all")

    def __init__(self):
        self.text = []
        self.bot = []
        self.log = []
        self.all = []

    @staticmethod
    def get_container(lvl, *text):
        # Распределяем на текст и class Counter
        tx = text[0]
        if isinstance(tx, Counter):
            tx = [tx]
        four = get_bit(lvl, 4) * 16
        if type(tx) in (tuple, list):
            for elem in tx:
                if isinstance(elem, Counter):
                    for _, _b, _t in elem.get_text(four):
                        yield _, _b, _t
                else:
                    yield -1, lvl, elem
        else:
            yield -1, lvl, tx

    @staticmethod
    def _is(lvl, arg, info):
        dif = arg & lvl & 14 or arg & info
        # if self.is_eq(lvl, arg) or (info and arg):
        if dif:
            return 1
        return 0

    def write_log(self, text, info=False):
        """
        Создаем буфер записей лога

        :param text: Передаваемый текст
        :param info: Вывод инфо=сообщений
        :return: None
        """
        _add_info = 16 if info else 0
        lvl = set_bit(0, 3)
        size = None
        txt = None
        is_size = False
        is_txt = False
        for zero, bit, elem in self.get_container(lvl, text):
            send_type = set_bit(0, self._is(bit, arguments.log, 16) * int_log)
            send_type |= set_bit(send_type, self._is(bit, arguments.console, 16) * int_console)
            send_type |= set_bit(send_type, self._is(bit, arguments.bot, 16) * int_bot)

            if zero == 2 and is_size is False:
                size = send_type, 'Перемещено или удалено: 0'
                is_size = True
            if zero == 1 and is_txt is False:
                txt = send_type, 'Всего обработано: 0'
                is_txt = True

            send_type = set_bit(0, self._is(bit, arguments.log, _add_info) * int_log)
            send_type |= set_bit(send_type, self._is(bit, arguments.console, _add_info) * int_console)
            send_type |= set_bit(send_type, self._is(bit, arguments.bot, _add_info) * int_bot)

            if send_type > 1:
                if zero == 1:
                    is_txt = True
                    txt = None
                if zero == 2:
                    is_size = True
                    size = None
                self.all.append((send_type, elem))
        if txt:
            self.all.append(txt)
        if size:
            self.all.append(size)

    def get_from_all(self, pos, forced):
        old = []
        if type(pos) is int:
            pos = [pos]
        for send_type, text in self.all:
            for num in pos:
                if (_s := get_bit(send_type, num)) or forced:
                    yield num, text
                    send_type ^= _s << num
            if send_type:
                old.append((send_type, text))

        self.all = old

    def send_all(self, sending=None, forced=False):
        """

        :param sending: Список устройств для вывода
        :param forced: Заставляет выводить информацию на устройство не указанное при задании write_log
        """
        send = set(sending or [int_log, int_console, int_bot]).intersection([int_log, int_console, int_bot])
        name = arguments.name
        _l = []
        _b = []
        for num, text in self.get_from_all(send, forced):
            if name and num == int_log:
                _l.append(f"{text}\n")
            if num == int_console:
                print(text)
            if num == int_bot:
                _b.append(f"{text}\n")

        if _l:
            save_log(_l, name)
        if _b:
            send_message(_b)
        self.all.clear()


class Counter:

    """
    Счетчик\n
    ----\n
    - total: Всего объектов в папке\n
    - folder: Количество папок\n
    - files: Файлы дошедшие до проверки include\n
    - exclude_files: Пропущено файлов\n
    - exclude_folders: Пропущено папок\n
    - move_object: Перемещенных объектов\n
    - delete_files: Удалено файлов\n
    - delete_folders: Удалено папок\n
    """
    # Ключи должны соответствовать аттрибутам класса
    counter_text = {"total": "Объектов", "folder": "Количество папок", "files": "Количество файлов",
                    "exclude_files": "Пропущено файлов", "exclude_folders": "Пропущено папок",
                    "delete_folders": "Удалено папок", "move_files": "Перемещено файлов",
                    "move_files_size": "Размер перемещенных файлов", "delete_files": "Удалено файлов",
                    "delete_files_size": "Размер освобожденного места"}

    __slots__ = tuple(counter_text.keys())

    __para__ = ("move_files", "move_files_size", "delete_files", "delete_files_size")

    def __init__(self):
        self.total = 0  # Всего объектов в папке
        self.folder = 0  # Количество папок
        self.files = 0  # Файлы дошедшие до проверки include
        self.exclude_files = 0  # Пропущено файлов
        self.exclude_folders = 0  # Пропущено папок
        self.delete_folders = 0  # Удалено папок
        self.move_files = 0  # Перемещенных объектов
        self.move_files_size = 0  # Объем перемещенных файлов
        self.delete_files = 0  # Удалено файлов
        self.delete_files_size = 0  # Объем освобожденного места

    def __iadd__(self, other):
        if type(self) == type(other):
            for key in self.counter_text:
                value = other.__getattribute__(key) + self.__getattribute__(key)
                self.__setattr__(key, value)
        else:
            raise TypeError(f"{type(self)} != {type(other)}")
        return self

    def get_text(self, four):
        message = Message(self)
        message.get_len()
        text = []
        it = iter(self.counter_text.items())
        for key, value in it:
            zero = 1
            vals = getattr(self, key)
            if vals > 0:
                level = set_bit(0, 2)
            else:
                level = set_bit(0, 3)
            level |= four
            text += (zero, level, f"{value: <{message.len_text}} : {vals}"),
            if key in self.__para__:
                zero = 2
                key_next, value_next = next(it)
                val = getattr(self, key_next)
                if val or vals:
                    level = set_bit(0, 1)
                else:
                    level = set_bit(0, 3)
                txt = f"{value_next:<{message.len_text2}} : {human_read_format(val)}"
                text += (zero, level, f"{text.pop()[2]:<{message.len_text + message.len_value + 5}}" + txt),
        if len(text) == 0:
            return set_bit(0, 3), 'В данной папке ничего не найдено!'
        return text

    def __len__(self):
        return 0


class Message:
    def __init__(self, namespace: Counter):
        self.messaging = []
        self.len_value = 0
        self.len_text = 0
        self.len_text2 = 0
        self.__namespace__ = namespace
        self.__size__ = self.__namespace__.__para__[1::2]

    def _get_len1(self):
        max_text = [(key, val := str(getattr(self.__namespace__, value)), len(key), len(val)) for value, key in
                    self.__namespace__.counter_text.items() if value not in self.__size__]
        self.len_value = max(max_text, key=lambda x: x[3])[3]
        self.len_text = max(max_text, key=lambda x: x[2])[2]

    def _get_len2(self):
        max_text = [(key, val := str(getattr(self.__namespace__, value)), len(key), len(val)) for value, key in
                    self.__namespace__.counter_text.items() if value in self.__size__]
        self.len_text2 = max(max_text, key=lambda x: x[2])[2]

    def get_len(self):
        self._get_len1()
        self._get_len2()


class Analyze:
    __slots__ = ("folders", "deep", "equals", "lock", "count", "rule")

    def __init__(self, files):
        """

        :type files: Path | Analyze
        """
        if isinstance(files, Analyze):
            self.folders = files.folders
            self.deep = files.deep
            self.lock = files.lock
            self.count = files.count
            self.rule = files.rule
            self.equals = change_parent_equal(files.equals)
        else:
            self.folders = Path(files)
            self.deep = []
            self.equals = {}
            self.lock = False
            self.count = get_count(self.folders)
            self.rule = {}

    def __repr__(self):
        return f"{self.folders.as_posix()!r}, deep={self.deep}, equals={self.equals}, lock={self.lock}\n{self.rule}\n"


def delete_folder(elem: Analyze):
    """Удаляем папку со всем содержимым"""
    if arguments.work:
        rmtree(elem.folders.as_posix(), ignore_errors=True)


def delete_files(elem: Analyze):
    """Удаляем файл"""
    if arguments.work:
        elem.folders.unlink(missing_ok=True)


def replace_files(elem: Analyze, old_dir: Path):
    """Перемещаем файл"""
    if arguments.work:
        if old_dir.exists() is False:
            old_dir.mkdir(parents=True)
        elem.folders.replace(old_dir.joinpath(elem.folders.name))


class Deleter:
    def __new__(cls, obj: Analyze, old_dir: Path, count: Counter, *args):
        if obj.folders.is_dir():
            ret = cls.work_folder(obj, old_dir, count)
        else:
            ret = cls.work_files(obj, old_dir, count)
        return ret

    @classmethod
    def get_size(cls, elem: Path, size=0, fil=0, fol=0):
        for files in elem.iterdir():
            if files.is_file():
                size += files.stat().st_size
                fil += 1
            else:
                fol += 1
                size, fil, fol = cls.get_size(files, size, fil, fol)
        return size, fil, fol

    @classmethod
    def work_folder(cls, elem: Analyze, old: Path, count: Counter):
        ss = elem.count.files - (elem.count.move_files + elem.count.delete_files) + \
             elem.count.folder - elem.count.delete_folders
        if (is_fast := elem.rule["@"]) or (ss == 0 and elem.lock is False):
            count.delete_folders += 1
            if is_fast:
                if ss > 0 and arguments.is_old is False:
                    txt = f"Переносим папку с содержимым: {elem.folders.as_posix()!r}"
                    elem.count.move_files_size, elem.count.files, elem.count.folder = cls.get_size(elem.folders)
                    count.move_files += elem.count.files
                    count.delete_folders += elem.count.folder
                    count.move_files_size += elem.count.move_files_size

                    replace_files(elem, old)
                    return txt, count
                txt = f"Удаление папки с содержимым: {elem.folders.as_posix()!r}"
                delete_folder(elem)
            else:
                txt = f"Удаляем папку: {elem.folders.as_posix()!r}"
                delete_folder(elem)
            return txt, count
        return False

    @classmethod
    def work_files(cls, elem: Analyze, old: Path, count: Counter):
        if (is_max := elem.rule["!"] is False) or elem.rule["+"]:
            if arguments.is_old:
                count.delete_files_size += elem.folders.stat().st_size
                count.delete_files += 1
                delete_folder(elem)
                return f"Удаляем файл: {elem.folders.as_posix()!r}", count
            count.move_files_size += elem.folders.stat().st_size
            count.move_files += 1
            replace_files(elem, old)
            if is_max:
                txt = f"Групповое перемещение: {elem.folders.as_posix()!r}"
            else:
                txt = f"Перемещаем файл: {elem.folders.as_posix()!r}"
            return txt, count
        return False


class FStat:
    """Обработка путей"""
    _lst_key: tuple

    def __init__(self, rule):

        self.max_time = 0
        self.parent_rule = Analyze(rule)
        self.count = self.parent_rule.count

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            err = traceback.format_tb(exc_tb)[0].split('\n')[:-1]
            mes = (f"-> {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} :: {self.parent_rule.folders}",
                   *err,
                   f"{exc_val}")

            logger.all = logger.all[-3:]    # выводим последние 3 строки
            logger.write_log(mes, info=True)
            logger.send_all()
        return False

    def add_parent_equal(self):
        for key, value in self.parent_rule.equals.items():
            if key in self._lst_key:
                continue
            self._lst_value += value,
            self._lst_key += key,

    def __enter__(self):
        # Читаем файл с правилами, если найден, добавляем правила по умолчанию.
        # На выходе получаем список кортежей с дельта-временем, определением можно ли удалить папку и именем этой папки
        rules = self.parent_rule.folders.joinpath(arguments.rule)
        rule_text = (rules.read_text(encoding="utf-8").splitlines() if rules.exists() else [])
        rule_text, deep = zip(*get_deep(rule_text, self.parent_rule.deep))
        self.parent_rule.deep = list(filter(len, deep))
        rule_text = list(filter(len, rule_text))
        rule_text += _add_znak(map(lambda x: f"{x}:0N", MAIN_EXCLUDE), "-")

        self._lst_key, self._lst_value = zip(*return_list(rule_text,
                                                          arguments.folder.joinpath(self.parent_rule.folders)))
        self.add_parent_equal()
        self._rules_compile_new = re.compile("(" + "$)|(".join(
            replace_template({"+": "[+]", "*": r".*", "!": "[!]", "@": "[@]", "?": "."}, self._lst_key)) + "$)")
        return self

    def _match_return(self, elem: str):
        znak = ["-", "@", "+", "!"]
        for z in znak:
            ret = self._rules_compile_new.match(z + elem)
            if ret is None or len(ret.string) == 0:
                continue
            keys = self._lst_key[ret.lastindex - 1]
            values = self._lst_value[ret.lastindex - 1]
            yield keys[0], values

    def sort(self, elem: Path):
        if elem.is_file():
            c_date = return_time_file(elem, MODIFICATION_TIME)
            self.max_time = max([self.max_time, c_date])
            return CURRENT_DATE - c_date
        else:
            return -1

    def get_bool_match(self, obj: Analyze) -> dict:
        exclude = delta("-", obj)
        plus = delta("+", obj)
        fast_folder = delta("@", obj)
        no_max = not delta("!", obj, self.max_time)
        return {"-": exclude, "+": plus, "!": no_max, "@": fast_folder}

    def get_info(self, files: Path) -> Analyze:
        obj = Analyze(files)
        obj.equals = dict(self._match_return(obj.folders.name))
        obj.lock = any([x[1] for x in obj.equals.values()])
        obj.rule = self.get_bool_match(obj)
        if obj.equals.get('@') is None and obj.equals:
            obj.count = get_count(obj.folders)
        return obj

    @staticmethod
    def work_rules(rules):
        preff = bool(rules.equals.get('@')) and rules.rule['@']
        if rules.folders.is_dir() and preff:
            return True
        if rules.equals.get('@'):
            return False
        return rules

    def list_folder(self, folders):
        for files in folders.iterdir():
            obj = self.get_info(files)
            mx = self.sort(files)
            yield mx, files, obj

    @property
    def iterdir(self):
        _log = []
        move_old_dir = arguments.trash_day.joinpath(self.parent_rule.folders.as_posix().replace(
            f"{arguments.folder.as_posix()}/", ""))

        for _, files, rules in sorted(self.list_folder(self.parent_rule.folders)):
            if rules.equals and rules.rule["-"] is False:
                rules.deep = self.parent_rule.deep
                if isinstance((ret := self.work_rules(rules)), Analyze) and files.is_dir():
                    yield ret
                elif ret is False:
                    continue
                if deleter := Deleter(rules, move_old_dir, self.count):
                    txt, self.count = deleter
                    _log.append(txt)
                continue

            self.count.exclude_files, self.count.exclude_folders = reduce(
                rules.folders, self.count.exclude_files, self.count.exclude_folders)

        logger.write_log([f"{' Поиск в: ' + f'{self.parent_rule.folders.as_posix()!r}' + ' ':-^{width_text}}",
                          *_log], info=True)
        logger.write_log(self.count)
        logger.send_all([int_log, int_console])


def recursive_dir(dir_name):
    count = Counter()
    with FStat(dir_name) as rules:
        for elem in rules.iterdir:
            count += recursive_dir(elem)
        count += rules.count
    return count


# Главный модуль
if __name__ == '__main__':
    main_time = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp())
    script_time = datetime.datetime.fromtimestamp(CURRENT_DATE)
    STR_NOW_DATE = datetime.datetime.fromtimestamp(CURRENT_DATE).strftime("%d-%m-%Y")
    arguments = alternative_parsers()
    create_log()
    total_count = Counter()
    total_parts = Counter()
    logger = Logger()
    logger.write_log(f"Current platform: {platform}")
    logger.write_log(f"{' Начато: ' + main_time.strftime('%d/%m/%Y, в: %H:%M') + ' ':+^{width_text}}", info=True)
    logger.write_log(f'Запуск осуществлен с параметром {arguments.command!r}. '
                     f'Файлы {"обрабатываются" if arguments.work else "не обрабатываются"}!')
    logger.send_all([int_log, int_console])
    for file in arguments.Search:
        arguments.is_old = file == arguments.trash
        if arguments.is_old:
            add_default_rule()
        if file.exists():
            total_parts = recursive_dir(file)
            logger.write_log(f"{' Итог: [' + f'{file.as_posix()!r}' + '] ':*^{width_text}}", info=True)
            logger.write_log(total_parts)
            total_count += total_parts
        else:
            logger.write_log(f"{file.as_posix()!r} заданная папка не найдена", info=True)
        logger.send_all([int_log, int_console])

    logger.write_log("#" * width_text)
    logger.write_log(f"{' Всего: ':-^{width_text}}", info=True)
    logger.write_log(total_count)
    tm_stop = datetime.datetime.now()
    logger.write_log(f"{' Закончено в: ' + tm_stop.strftime('%d/%m/%Y %H:%M') + ' ':+^{width_text}}", info=True)
    tz = datetime.timezone(datetime.timedelta(hours=0))
    sh = "%H: час., %M: мин., %S: сек., %f: мкс."
    logger.write_log(f"{'Время прогона':<25}:  "
                     f"{datetime.datetime.fromtimestamp((tm_stop - main_time).total_seconds(), tz=tz).strftime(sh)}",
                     info=True)

    logger.write_log(f"{'Время выполнения скрипта':<25}:  "
                     f"{datetime.datetime.fromtimestamp((tm_stop - script_time).total_seconds(), tz=tz).strftime(sh)}",
                     info=True)
    logger.write_log(f"Командная строка: {Path(__file__).absolute().as_posix()} {' '.join(argv[1:])}")

    logger.send_all()
