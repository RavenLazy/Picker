#  coding: utf-8

# -----------------------------------------------------------------------------------
# Формат записи правил:
# Пример: исключить папку <Audio>, если она создана не ранее 3 дней от текущей
# даты, если папка пуста не удалять. Пропустить файл <error.log>.
# Проверить папку <AudioBooks> и все файлы внутри если они старше 10 дней. Пустые
# папки - удалить
# Файл log.log перенести!
# Запись: каждое правило на своей строке
# Допустимые указания времени:
# Y - год
# M - месяц
# D - день
# H - час
#
# -Audio:3D:L
# !AudioBooks:10D:U или +AudioBooks:10D:U
# -error.log:0N   # 0 - текущая дата
# +log.log    # время не уаказано, поэтому от начвла времен :)
#
# -:  Пропускаем объект. Если указано время, то проверяем и
#     пропускаем только те, что старше.
#     Имеет приоритет над <@>
#     То есть проверяется первым!
#
# +:  Включаем этот объект, если времени прошло больше чем задано.
#     Приоритета не имеет
#
# !:  То-же, что и +, с одним исключением: время считается по самому "молодому" файлу
#     Работает только с файлами.
#     Имеет приоритет над <+>
#
# @:  "Быстрая" проверка. Только для папок. Если время папки старше, чем задано, то
#     папка удаляется со всем содержимым. Рекомендуется для папок содержащих данные
#     относящиеся к одному объекту или приложению. Например, папка с перемещенными
#     файлами или установленная программа.
#     Имеет приоритет над <!>
# -----------------------------------------------------------------------------------

import sys
import re
import datetime
import pathlib
import argparse
import shutil

CREATION_TIME = 0
MODIFICATION_TIME = 1
ACCESS_TIME = 2

NAME_MOVE_DIR = "Trash"
DAY_TO_PURGE = 14
Default_rule_old_days = f"@*:1D:U\n-*:{DAY_TO_PURGE}D:U"

NAME_RULE = ".rule"
NAME_PATH = ".path"
ERROR_LOG = "error.log"

START_FOLDER_NAME = pathlib.Path.cwd()
CURRENT_DATE = datetime.datetime.now().replace(second=0, microsecond=0).timestamp()

MAIN_EXCLUDE = [NAME_RULE, NAME_PATH]
DELETE_ALL_OLD = False

DEFAULT_LOG = "log.log"

list_includes_znak = ("@", "!", "+")

compile_rule = re.compile(r"[:]")
old_date_pattern = re.compile(r"\d{2}-\d{2}-\d{4}$")

counter_text = ("Объектов: ", "Количество папок: ", "Учтенных файлов: ", "Пропущено: ",
                "Перемещено: ", "Удалено файлов: ", "Удалено папок: ")


def old_default_rule(path: pathlib.Path):
    rule = path.joinpath(NAME_RULE)
    if path.exists() is False:
        path.mkdir(parents=True)
    if rule.exists():
        return None
    with rule.open("w") as file:
        file.writelines(Default_rule_old_days)
    return None


def add_in_list_set(array: list, text: str):
    if text not in array:
        return array + [text]
    return array


def return_list(text, folders):
    tmp = []
    for name, times, lock in decompress(text, folders):
        if name in tmp:
            continue
        tmp.append(name)
        yield name, (times, lock)


def return_list_main(elem: argparse.Namespace):
    for item in elem.__dict__.values():
        if type(item) == list and len(item) > 0:
            vv = []
            for values in item:
                if type(values) == list:
                    vv.extend(values)
                else:
                    yield values
            if vv:
                yield vv
            continue
        yield item


def for_size(item):
    ident: str
    size: int
    coefficient = 1000
    size, ident = re.findall(r"([\d]+)([MmKkBb]?)", item)[0]
    size = int(size)
    return {"m": size * (coefficient ** 2), "k": size * coefficient, "b": size * 1}.get(ident.lower(), size)


def parse_arg():
    arg = argparse.ArgumentParser(description="Перенос старых файлов в отдельную директорию с сохранением путей",
                                  prog="Trash")
    group1 = arg.add_mutually_exclusive_group()
    group2 = arg.add_mutually_exclusive_group()

    arg.add_argument("--parent", "-p", nargs=1, default=START_FOLDER_NAME, metavar=f"{START_FOLDER_NAME.as_posix()!r}",
                     type=pathlib.Path, help=f"Откуда начинаем искать. По умолчанию папка запуска скрипта.")

    group1.add_argument("--pathname", "-pn", type=str, default=NAME_PATH, nargs=1, metavar=".path",
                        help=f"Имя файла с папками для поиска. По умолчанию: {NAME_PATH!r}")

    arg.add_argument("--rule", "-r", type=str, default=NAME_RULE, nargs=1, metavar=".rule",
                     help=f"Имя файла с правилами анализа: По умолчанию: {NAME_RULE!r}")

    # arg.add_argument("--delete_old", '-d', action="store_true", default=DELETE_ALL_OLD,
    #                  help=f"Удаляем все старые файлы из папки переноса. Анализируется только основная папка. "
    #                       f"По умолчанию, анализируется каждый файл.")

    arg.add_argument("--name_old", "-n_o", nargs=1, type=str, default=NAME_MOVE_DIR,
                     metavar=f"{NAME_MOVE_DIR!r}",
                     help=f"Имя папки в которую будут переносится старые файлы. По умолчанию: {NAME_MOVE_DIR!r}")

    group1.add_argument("-s", "--search", type=str, nargs="+", action="append",
                        metavar="'Folder/1' 'Folder/2' ...", default=[], help="Папки поиска")

    group2.add_argument("-l", "--log", action="store_true",
                        help=f"Разрешить запись лога. Задается без дополнительного аргумента. По умолчанию лог не "
                             f"пишем. Если разрешена запись, то пишем в {DEFAULT_LOG!r}")

    group2.add_argument("-lf", "--log_file", action="store", metavar="log_file.log",
                        help="Имя файла для записи лога. Если указано будем писать в этот файл. "
                             "Указывается либо имя файла, либо параметр '-l (--log)'")

    arg.add_argument("-al", "--append_log", action="store_false",
                     help="Начинать писать лог файл заново. По умолчанию: дописывать")

    arg.add_argument("-sl", "--size_log", type=for_size, default=1000, metavar='1000',
                     help="Размер лог файла в байтах. При превышении файл будет переименован и создан новый. "
                          "По умолчанию: 1000 байт. ")

    arg.add_argument("-o", "--out", action="store_false",
                     help="Выводить информацию в консоль? По умолчанию: Да")

    arg.add_argument("-V", "--version", action="version", version="%(prog)s 1.0")

    arg_s = arg.parse_args()
    start_path: pathlib.Path

    start_path, name_path_file, name_rule_file, name_trash_dir, search_path, is_log_write, \
        name_log, is_append_log, log_size, is_console_out = return_list_main(arg_s)

    search_path = add_in_list_set(search_path, name_trash_dir)
    if any([x.count(':') or x.count('/') for x in search_path]):
        raise ValueError('Путь должен быть относительным. Без слэш и двоеточий')

    if start_path.exists() is False:
        raise OSError(f"{START_FOLDER_NAME.as_posix()!r} not exists")

    if is_log_write or name_log:
        name_log = pathlib.Path(name_log or DEFAULT_LOG)

    name_path = search_path or name_path_file
    ss = []
    if search_path:
        ss = [elem[0] if type(elem) == list else elem for elem in name_path]
    else:
        _start = start_path.joinpath(name_path)
        if _start.exists():
            ss = _start.read_text().splitlines()

    ss = map(start_path.joinpath, ss)

    return start_path, ss, name_rule_file, name_trash_dir, name_log, is_append_log, log_size, is_console_out


def log():
    if append_log:
        if NAME_LOG.stat().st_size >= size_log:
            name = NAME_LOG.as_posix().split(".")[0]
            fullname = pathlib.Path('-'.join([name, STR_NOW_DATE + NAME_LOG.suffix]))
            count = 0
            while fullname.exists():
                count += 1
                fullname = pathlib.Path(
                    '-'.join([name, STR_NOW_DATE + f"({str(count).zfill(3)}){NAME_LOG.suffix}"]))
            NAME_LOG.replace(fullname)
    else:
        NAME_LOG.unlink(missing_ok=True)


def write_log(text, n_log=None):
    if n_log is None:
        n_log = NAME_LOG
    lines = type(text) == list
    #  Выводим текст в консоль
    if console_out:
        if lines:
            print(*text, sep="\n")
        else:
            print(text)

    #  Выводим текст в файл
    if n_log:
        with open(n_log, "a") as file:
            if lines:
                file.writelines([f"{line}\n" for line in text])
            else:
                file.write(text + '\n')


def round_time(x: datetime.datetime.second):
    hour_in_seconds = 60 * 60
    half_hour_in_seconds = 60 * 30
    if x % hour_in_seconds > half_hour_in_seconds:
        return ((x // hour_in_seconds) + 1) * hour_in_seconds
    else:
        return (x // hour_in_seconds) * hour_in_seconds


def return_time_file(name: pathlib.Path, type_time):
    return {ACCESS_TIME: name.stat().st_atime, MODIFICATION_TIME: name.stat().st_mtime,
            CREATION_TIME: name.stat().st_ctime}[type_time]


def revert_rules(elem: dict):
    def _mn(x):
        for d, c in zip(x, ["Y:U", "M:U", "D:U", "H:U"]):
            if d == 0:
                continue
            yield d, c
    # Переводим секунды в корректное время
    if len(elem) == 0:
        return []

    for znak, (seconds, lock) in elem.items():
        _n, _d = min(_mn([seconds // 31536000, seconds // 2592000, seconds // 86400, seconds // 3600]),
                     default=(0, "H:U"))
        _ret = f"{znak}*:{_n}{_d}"
        yield _ret


def difference_date(date):
    # Разница между текущим и заданным временем.
    return CURRENT_DATE - date


def replace_template(pat, item):
    # Подменяем знаки из item в знаки из pat
    items = item
    for elem in items:
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


def decompress(el: list, folders: pathlib.Path):
    # Раскладываем время из файла правил
    count = 0
    for item in el:
        count += 1
        pat = compile_rule.split(item, maxsplit=2)
        pat += [''] * (3 - len(pat))
        name, dates, lock = pat
        dates = dates or '0N'
        try:
            dates = int(normal_date(dates)) or (0 if name[0] in list_includes_znak else int(CURRENT_DATE))
        except ValueError:
            write_log(f"{item!r} ошибка разметки в списке правил {NAME_RULE!r}."
                      f"\nПапка: {folders.as_posix()!r}.\nСтрока: {count}")
            sys.exit(1)
        lock = False if lock == "U" else True  # True - папка не будет удалена
        yield name, dates, lock


class Counter:
    """
    Счетчик\n
    ----\n
    - total: Всего объектов в папке\n
    - folder: Количество папок\n
    - files: Файлы дошедшие до проверки include\n
    - exclude: Пропущено файлов\n
    - move_object: Перемещенных объектов\n
    - delete_files: Удалено файлов\n
    - delete_folders: Удалено папок\n
    """

    __slots__ = ("total", "folder", "files", "exclude", "move_object", "delete_files", "delete_folders")

    def __init__(self):
        self.total = 0  # Всего объектов в папке
        self.folder = 0  # Количество папок
        self.files = 0  # Файлы дошедшие до проверки include
        self.exclude = 0  # Пропущено файлов
        self.move_object = 0  # Перемещенных объектов
        self.delete_files = 0  # Удалено файлов
        self.delete_folders = 0  # Удалено папок

    def __iadd__(self, other):
        if type(self) == type(other):
            for key in self.__slots__:
                value = other.__getattribute__(key) + self.__getattribute__(key)
                self.__setattr__(key, value)
        else:
            raise TypeError(f"{type(self)} != {type(other)}")
        return self

    def __str__(self):
        return '\n'.join(
            [f"{key}{value}" for key, value in zip(counter_text, [self.__getattribute__(x) for x in self.__slots__])])


class Job:
    __slots__ = ("filename", "equals", "move_dir", "is_max", "type_time", "max_time", "counter", "log")

    def __init__(self, lots):
        if isinstance(lots, Job):
            self.filename = lots.filename
        else:
            self.filename = pathlib.Path(lots)
        self.equals = {}
        self.move_dir: str | pathlib.Path = ''
        self.is_max = False
        self.max_time = CURRENT_DATE
        self.type_time: int = MODIFICATION_TIME
        self.counter = Counter()
        self.log = []

    def __str__(self):
        return self.filename.as_posix()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.filename!r})"

    @staticmethod
    def del_or_move(foo, *args, **kwargs):
        foo(*args, **kwargs)
        return 1

    def _delta(self, znak):
        fl = self.equals.get(znak, False)
        if fl is False:
            return {znak in list_includes_znak: False, znak == "-": True}[True]
        date, _ = fl
        if znak == "!":
            return int(CURRENT_DATE - self.max_time) > date

        return int(CURRENT_DATE - return_time_file(self.filename, self.type_time)) > date

    def exclude(self):
        ret = not self._delta("-")
        if ret:
            self.counter.exclude += 1
        return ret

    def include(self):
        self.counter.files += 1
        return self._delta("!") or self._delta("+")

    def is_dir(self):
        if self.filename.is_dir():
            self.counter.folder += 1
            return True
        return False

    def work_file(self):
        f"""
        Переносим файл в папку Old {NAME_MOVE_DIR}
        :return: bool
        """
        move_files = self.move_dir.joinpath(self.filename.name)
        self.log.append(f"Время {fullpath.joinpath(self.filename.name).as_posix()!r} вышло за заданный диапазон. "
                        f"Переносим в {move_files.as_posix()!r}")
        if not self.move_dir.exists():
            self.move_dir.mkdir(parents=True)
        self.counter.move_object += self.del_or_move(self.filename.replace, move_files)
        return True

    def _is_lock(self):
        lock: str
        _, lock = zip(*self.equals.values())
        if lock[0].lower() == 'l':
            return True
        return False

    def del_dir(self):
        """
        Удаляем папку
        """
        self.log.append(f'Удаляем папку {self.filename}')
        self.counter.delete_folders += self.del_or_move(shutil.rmtree, self.filename.as_posix(), ignore_errors=True)

    def is_fast_date(self):
        fast = self._delta("@")
        return fast

    def is_fast(self):
        fast = self.equals.get("@") is not None and self._is_lock() is False
        return fast

    def empty(self):
        if self.filename.is_dir():
            full = any((x for x in self.filename.iterdir())) or self._is_lock()
            if full:
                return False
            self.del_dir()
            return True


class JobOld(Job):

    def work_file(self):
        """
        Удаляем файл или папку.
        """
        self.log.append(f"{self.filename.as_posix()!r} Время хранения истекло - удаляем!")
        if self.filename.is_file():
            self.counter.delete_files += self.del_or_move(self.filename.unlink, missing_ok=True)
        else:
            self.counter.delete_folders += self.del_or_move(self.filename.rmdir)
        return True


class FStat:
    """Обработка путей"""

    def __init__(self, rule: Job | str, type_time=MODIFICATION_TIME):
        if IS_OLD:
            self._lot = JobOld(rule)
        else:
            self._lot = Job(rule)

        self._lot.type_time = type_time
        self.is_max = False
        if isinstance(rule, Job):
            self.directory = pathlib.Path(rule.filename)
            self._parent = revert_rules(rule.equals)
            self.is_max = len(rule.equals.get("!", [])) > 0
        else:
            self.directory = pathlib.Path(rule)
            self._parent = []
        self._lot.move_dir = MOVE_OLD / self.directory.as_posix().replace(f"{START_FOLDER_NAME.as_posix()}/", "")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            write_log([exc_type, exc_tb, exc_val], NAME_LOG or ERROR_LOG)
            return False
        txt = ' Ищу в: ' + f"{self.directory.as_posix()!r} "
        write_log(f"{txt:-^100}")
        if self._lot.log:
            write_log(self._lot.log)
            write_log('=' * 100)
        write_log(self._lot.counter)

    def get_counter(self):
        return self._lot.counter

    @staticmethod
    def _add_znak(item, znak):
        return list(map(lambda x: znak + x if x[0] not in list_includes_znak else x, item))

    def __enter__(self):
        # Читаем файл с правилами, если найден, добавляем правила по умолчанию.
        # На выходе получаем список кортежей с дельта-временем, определением можно ли удалить папку и именем этой папки

        rules = self.directory.joinpath(NAME_RULE)
        rule_text = (rules.read_text(encoding="utf-8").splitlines() if rules.exists() else [])
        rule_text += self._add_znak(map(lambda x: f"{x}:0N", MAIN_EXCLUDE), "-") + self._add_znak(self._parent, "+")
        self._lst_key, self._lst_value = zip(*return_list(rule_text, START_FOLDER_NAME.joinpath(self.directory)))
        self._rules_compile_new = re.compile("(" + "$)|(".join(
            replace_template({"+": "[+]", "*": r".+", "!": "[!]", "@": "[@]"}, self._lst_key)) + "$)")
        return self

    @staticmethod
    def _get_item(elem, last, index=0):
        return elem[last.lastindex - 1][index]

    def _match_return(self, elem):
        znak = ["-", "@", "+", "!"]
        for z in znak:
            ret = self._rules_compile_new.match(z + elem)
            if ret is None:
                continue
            yield self._get_item(self._lst_key, ret), (
                self._get_item(self._lst_value, ret),
                {True: "L", False: "U"}.get(self._get_item(self._lst_value, ret, 1), True))

    def iterdir(self):
        if self.is_max:
            self._lot.max_time = max([return_time_file(x, self._lot.type_time)
                                      for x in self.directory.iterdir() if x.is_file()] or [CURRENT_DATE])

        for self._lot.filename in sorted(self.directory.iterdir(), key=lambda x: x.is_file()):
            self._lot.counter.total += 1
            self._lot.equals = dict(self._match_return(self._lot.filename.name))
            if self._lot.equals:
                yield self._lot


def recursive_dir(dir_name):
    count = Counter()
    with FStat(dir_name) as rules:
        elem: Job
        for elem in rules.iterdir():
            if elem.exclude():
                continue
            if elem.is_dir():
                if elem.is_fast():
                    if elem.is_fast_date():
                        elem.del_dir()
                    continue
                count += recursive_dir(elem)
                elem.empty()
                continue
            if elem.include():
                elem.work_file()

        count += rules.get_counter()

    return count


if __name__ == '__main__':
    # Todo: Требуется окончательная проверка.
    #  Проверить все глобальные переменные. DELETE_ALL_OLD - вроде и не нужно! Проверить остальные!

    START_FOLDER_NAME, folder, NAME_RULE, NAME_MOVE_DIR, \
        NAME_LOG, append_log, size_log, console_out = parse_arg()
    STR_NOW_DATE = datetime.datetime.fromtimestamp(CURRENT_DATE).strftime("%d-%m-%Y")
    try:
        if NAME_LOG.exists():
            log()
    except AttributeError:
        pass
    tm = datetime.datetime.fromtimestamp(CURRENT_DATE)
    write_log(f"{' '.join([' Start scan at:', tm.strftime('%d-%m-%Y %H:%M'), ' ']):-^100}")
    write_log(f"Current platform: {sys.platform}")
    MOVE_MAIN_OLD = START_FOLDER_NAME.joinpath(NAME_MOVE_DIR)
    old_default_rule(MOVE_MAIN_OLD)
    MOVE_OLD = MOVE_MAIN_OLD.joinpath(STR_NOW_DATE)
    total_count = Counter()
    total_parts = Counter()

    for l_p in folder:
        fullpath = START_FOLDER_NAME.joinpath(l_p)
        if l_p.exists():
            IS_OLD = l_p.name == MOVE_MAIN_OLD.name
            total_parts = recursive_dir(l_p)
            s = ' '.join(['*', '[\'' + fullpath.as_posix() + '\']', '*'])
            write_log(f"{s:*^100}\n{total_parts}")
            total_count += total_parts
        else:
            write_log(f"{fullpath.as_posix()!r} заданная папка не найдена")

    write_log(f"{'#' * 100}\n{' Всего: ':-^100}\n{total_count}")
    tm_stop = datetime.datetime.now()
    tz = datetime.timezone(datetime.timedelta(hours=0))
    write_log(f"{' '.join([' Stop scan at:', tm_stop.strftime('%d-%m-%Y %H:%M'), ' ']):-^100}")
    write_log(datetime.datetime.fromtimestamp((tm_stop - tm).total_seconds(),
                                              tz=tz).strftime("Часов: %H Минут: %M Секунд: %S Микросекунд: %f"))
