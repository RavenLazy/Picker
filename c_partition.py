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

import mlist as mlist

CREATION_TIME = 0
MODIFICATION_TIME = 1
ACCESS_TIME = 2

NAME_MOVE_DIR = "Trash"
DAY_TO_PURGE = 14
Default_rule_old_days = f"@*:1D:U\n-*:{DAY_TO_PURGE}D:U"

NAME_RULE = ".rule"
NAME_PATH = ".path"
ERROR_LOG = "error.log"



CURRENT_DATE = datetime.datetime.now().replace(second=0, microsecond=0).timestamp()

MAIN_EXCLUDE = [NAME_RULE, NAME_PATH]
DELETE_ALL_OLD = False

DEFAULT_LOG = "log.log"

list_includes_znak = ("@", "!", "+")

compile_rule = re.compile(r"[:]")
old_date_pattern = re.compile(r"\d{2}-\d{2}-\d{4}$")

counter_text = ("Объектов", "Количество папок", "Количество файлов", "Пропущено файлов", "Пропущено папок",
                "Перемещено", "Удалено файлов", "Удалено папок")


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
    """Создаем список с неповторяющимися элементами"""
    if text not in array:
        return array + [text]
    return array


def decompress(el: list, folders: pathlib.Path, dp: bool):
    # Раскладываем время из файла правил
    name: str
    count = 0
    for item in el:
        count += 1
        pat = compile_rule.split(item, maxsplit=3)
        pat += [''] * (4 - len(pat))
        name, dates, lock, deep = pat
        deep = int(deep or '0')
        dates = dates or '0N'
        if dp is False:
            try:
                dates = int(normal_date(dates)) or (0 if name[0] in list_includes_znak else int(CURRENT_DATE))
            except ValueError:
                write_log(f"{item!r} ошибка разметки в списке правил {NAME_RULE!r}."
                          f"\nПапка: {folders.as_posix()!r}.\nСтрока: {count}")
                sys.exit(1)
            lock = False if lock == "U" else True  # True - папка не будет удалена

        yield name, (dates, lock, deep)


def return_list(text, folders, deep=False):
    tmp = []
    for name, args in decompress(text, folders, deep):
        if name in tmp:
            continue
        tmp.append(name)
        yield name, args


def return_list_main(elem: argparse.Namespace):
    """Для создания списка из аргументов командной строки"""
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


def create_path(namespace, values):
    folder = getattr(namespace, 'folder')
    for f in folder:
        for v in values:
            r = f.joinpath(v)
            yield r


def flatten(values):
    s = []
    for x in values:
        s.extend(x)
    return s


def read_path(namespace, values):
    for files in create_path(namespace, [values]):
        if files.exists():
            with open(files, 'r') as f:
                yield create_path(namespace, f.read().splitlines())
    return None


class ActionFile(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))
        setattr(namespace, 'Search', flatten(read_path(namespace, values)))


class ActionSearch(ActionFile):

    def __call__(self, parser, namespace, values: list, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))

        values = self.create_path(namespace, values)
        if atr := getattr(namespace, 'Search'):
            if type(atr) == list:
                _, values = atr.extend(values), atr
        setattr(namespace, self.dest, list(values))


def alternative_parsers():
    # Todo: Добавить обработку .rule.
    #  Продумать обработку rule, search с помощью класса Action

    arg = argparse.ArgumentParser(description="Перенос старых файлов в отдельную директорию с сохранением путей",
                                  prog="Trash")

    search_folder = arg.add_mutually_exclusive_group()

    arg.add_argument("command", metavar="<execute> | <test>", type=str, choices=['execute', 'test'],
                     help="Выполнение или тестовый прогон.")

    arg.add_argument("-folder", nargs='*', metavar="<start folder name>", required=True, action='extend',
                     type=pathlib.Path, help=f"Откуда начинаем искать")

    search_folder.add_argument("-Search", type=str, action=ActionSearch,
                               metavar="<Folder1> <Folder2> ... <FolderNNN>", nargs='*',
                               help=f"Папки поиска.")
    search_folder.add_argument("-pathname", type=str, default='.path', metavar=".path", action=ActionFile,
                               help=f"Имя файла с папками для поиска. По умолчанию: <%(default)s>")

    arg.add_argument("-rule", default='.rule', action=ActionFile)
    arg.add_argument("-trash", nargs=1, type=str, default='Trash',
                     metavar="<'Trash'>",
                     help=f"Имя папки в которую будут переносится старые файлы. По умолчанию: '%(default)s'")

    group_log = arg.add_subparsers(help='Лог файл')
    parser_a = group_log.add_parser("log", help="Help: log -h")
    parser_a.add_argument("-name", default=DEFAULT_LOG, help=f"Имя лог-файла <{DEFAULT_LOG}>")
    parser_a.add_argument("-size", default=1000, type=for_size, help="Размер лога <1000>")
    parser_a.add_argument("-append", action="store_true", help="Продолжать дописывать <True>")

    arg.add_argument("-console", action="store_true", help="Выводить в консоль. По умолчанию <False>")

    arg.add_argument('--version', "-V", action='version', version='%(prog)s 1.0')

    # Надо ли писать в лог файл
    logging = type((arguments := arg.parse_args().__dict__).get("size")) == int

    # Создаем список папок для поиска. Читаем файл .path или что задано в pathname
    if arguments['Search'] is None:
        arguments['Search'] = flatten(read_path(arg.parse_args(), arguments['pathname']))

    print(arguments, logging, sep='\n')
    return arguments, logging


# def parse_arg():
#     # Todo: Пересмотреть аргументы командной строки
#
#     arg = argparse.ArgumentParser(description="Перенос старых файлов в отдельную директорию с сохранением путей",
#                                   prog="Trash")
#     group1 = arg.add_mutually_exclusive_group()
#     group2 = arg.add_mutually_exclusive_group()
#
#     arg.add_argument("--parent", "-p", nargs=1, default=START_FOLDER_NAME, metavar=f"{START_FOLDER_NAME.as_posix()!r}",
#                      type=pathlib.Path, help=f"Откуда начинаем искать. По умолчанию папка запуска скрипта.")
#
#     group1.add_argument("--pathname", "-pn", type=str, default=NAME_PATH, nargs=1, metavar=".path",
#                         help=f"Имя файла с папками для поиска. По умолчанию: {NAME_PATH!r}")
#
#     arg.add_argument("--rule", "-r", type=str, default=NAME_RULE, nargs=1, metavar=".rule",
#                      help=f"Имя файла с правилами анализа: По умолчанию: {NAME_RULE!r}")
#
#     arg.add_argument("--name_old", "-n_o", nargs=1, type=str, default=NAME_MOVE_DIR,
#                      metavar=f"{NAME_MOVE_DIR!r}",
#                      help=f"Имя папки в которую будут переносится старые файлы. По умолчанию: {NAME_MOVE_DIR!r}")
#
#     group1.add_argument("-s", "--search", type=str, nargs="+", action="append",
#                         metavar="'Folder/1' 'Folder/2' ...", default=[], help="Папки поиска")
#
#     group2.add_argument("-l", "--log", action="store_true",
#                         help=f"Разрешить запись лога. Задается без дополнительного аргумента. По умолчанию лог не "
#                              f"пишем. Если разрешена запись, то пишем в {DEFAULT_LOG!r}")
#
#     group2.add_argument("-lf", "--log_file", action="store", metavar="log_file.log",
#                         help="Имя файла для записи лога. Если указано будем писать в этот файл. "
#                              "Указывается либо имя файла, либо параметр '-l (--log)'")
#
#     arg.add_argument("-al", "--append_log", action="store_false",
#                      help="Начинать писать лог файл заново. По умолчанию: дописывать")
#
#     arg.add_argument("-sl", "--size_log", type=for_size, default=1000, metavar='1000',
#                      help="Размер лог файла в байтах. При превышении файл будет переименован и создан новый. "
#                           "По умолчанию: 1000 байт. ")
#
#     arg.add_argument("-o", "--out", action="store_false",
#                      help="Выводить информацию в консоль? По умолчанию: Да")
#
#     arg.add_argument("-V", "--version", action="version", version="%(prog)s 1.0")
#
#     arg_s = arg.parse_args()
#     start_path: pathlib.Path
#
#     start_path, name_path_file, name_rule_file, name_trash_dir, search_path, is_log_write, \
#         name_log, is_append_log, log_size, is_console_out = return_list_main(arg_s)
#
#     search_path = add_in_list_set(search_path, name_trash_dir)
#     if any([x.count(':') or x.count('/') for x in search_path]):
#         raise ValueError('Путь должен быть относительным. Без слэш и двоеточий')
#
#     if start_path.exists() is False:
#         raise OSError(f"{START_FOLDER_NAME.as_posix()!r} not exists")
#
#     if is_log_write or name_log:
#         name_log = pathlib.Path(name_log or DEFAULT_LOG)
#
#     name_path = search_path or name_path_file
#     ss = []
#     if search_path:
#         ss = [elem[0] if type(elem) == list else elem for elem in name_path]
#     else:
#         _start = start_path.joinpath(name_path)
#         if _start.exists():
#             ss = _start.read_text().splitlines()
#
#     ss = map(start_path.joinpath, ss)
#
#     return start_path, ss, name_rule_file, name_trash_dir, name_log, is_append_log, log_size, is_console_out


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
    if len(text) > 0:
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
        return {}
    for znak, (seconds, lock, deep) in elem.items():
        _n, _d = min(_mn([seconds // 31536000, seconds // 2592000, seconds // 86400, seconds // 3600]),
                     default=(0, "H:U"))
        yield f"{znak}*:{_n}{_d}"


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


def get_count(elem: pathlib.Path):
    ret = Counter()
    if elem.is_dir():
        ret.total = len(obj := [(files, files.is_file()) for files in elem.iterdir()])
        ret.files = len([x for x in obj if x[1]])
        ret.folder = ret.total - ret.files
    else:
        ret.total = 1
        ret.files = 1
    return ret


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

    __slots__ = ("total", "folder", "files", "exclude_files", "exclude_folders", "move_object",
                 "delete_files", "delete_folders")

    def __init__(self):
        self.total = 0  # Всего объектов в папке
        self.folder = 0  # Количество папок
        self.files = 0  # Файлы дошедшие до проверки include
        self.exclude_files = 0  # Пропущено файлов
        self.exclude_folders = 0  # Пропущено папок
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
            [f"{key.ljust(18):s}:  {value:d}" for key, value in
             zip(counter_text, [self.__getattribute__(x) for x in self.__slots__])])

    def __len__(self):
        return len(self.__slots__)


class Analyze:

    __slots__ = ("folders", "deep", "equals", "lock", "count", "rule")

    def __init__(self, files):
        """

        :type files: pathlib.Path | Analyze
        """
        if isinstance(files, Analyze):
            self.folders = files.folders
            self.deep = files.deep
            self.equals = revert_rules(files.equals)
            self.lock = files.lock
            self.count = files.count
            self.rule = files.rule
        else:
            self.folders = pathlib.Path(files)
            self.deep = []
            self.equals = {}
            self.lock = False
            self.count = get_count(self.folders)
            self.rule = {}

    def __repr__(self):
        return f"{self.folders.as_posix()!r}, deep={self.deep}, equals={self.equals}, lock={self.lock}\n{self.rule}\n"


def delta(znak, elem: Analyze | dict, max_time=None):
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


class Deleter:
    def __new__(cls, obj: Analyze, old_dir: pathlib.Path, count: Counter, *args):
        if obj.folders.is_dir():
            ret = cls.work_folder(obj, count)
        else:
            ret = cls.work_files(obj, old_dir, count)
        return ret

    @classmethod
    def work_folder(cls, elem: Analyze, count: Counter):
        ss = elem.count.files - (elem.count.move_object + elem.count.delete_files) + \
             elem.count.folder - elem.count.delete_folders
        if (is_fast := elem.rule["@"]) or (ss == 0 and elem.lock is False):
            fast_deleter(elem)
            count.delete_folders += 1
            if is_fast:
                txt = f"Удаление папки с содержимым: {elem.folders.as_posix()!r}"
            else:
                txt = f"Удаляем папку: {elem.folders.as_posix()!r}"
            return txt, count
        return False

    @classmethod
    def work_files(cls, elem: Analyze, old: pathlib.Path, count: Counter):
        if (is_max := elem.rule["!"] is False) or elem.rule["+"]:
            if IS_OLD:
                delete(elem)
                count.delete_files += 1
                return f"Удаляем файл: {elem.folders.as_posix()!r}", count
            replace(elem, old)
            count.move_object += 1
            if is_max:
                txt = f"Групповое перемещение: {elem.folders.as_posix()!r}"
            else:
                txt = f"Перемещаем файл: {elem.folders.as_posix()!r}"
            return txt, count
        return False


def fast_deleter(elem: Analyze):
    """Удаляем папку со всем содержимым"""
    # shutil.rmtree(elem.folders.as_posix(), ignore_errors=True)
    pass


def delete(elem: Analyze):
    """Удаляем файл"""
    # elem.folders.unlink(missing_ok=True)
    pass


def replace(elem: Analyze, old_dir: pathlib.Path):
    """Перемещаем файл"""
    if old_dir.exists() is False:
        old_dir.mkdir(parents=True)
    # elem.folders.replace(old_dir)


class FStat:
    """Обработка путей"""

    def __init__(self, rule: Analyze | str):

        self.max_time = 0
        self.parent_rule = Analyze(rule)
        self.count = self.parent_rule.count

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            write_log([exc_type, exc_tb, exc_val], NAME_LOG or ERROR_LOG)
            return False

    @staticmethod
    def _add_znak(item, znak):
        return list(map(lambda x: znak + x if x[0] not in list_includes_znak else x, item))

    @staticmethod
    def get_deep(rule_text, dp):
        tmp = []
        rule_text += [elem for elem in dp]
        if len(rule_text) == 0:
            yield '', ''
        for name, args in decompress(rule_text, START_FOLDER_NAME, True):
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

    def __enter__(self):
        # Читаем файл с правилами, если найден, добавляем правила по умолчанию.
        # На выходе получаем список кортежей с дельта-временем, определением можно ли удалить папку и именем этой папки

        rules = self.parent_rule.folders.joinpath(NAME_RULE)
        rule_text = (rules.read_text(encoding="utf-8").splitlines() if rules.exists() else [])
        rule_text, deep = zip(*self.get_deep(rule_text, self.parent_rule.deep))
        self.parent_rule.deep = list(filter(len, deep))
        rule_text = list(filter(len, rule_text))
        rule_text += self._add_znak(map(lambda x: f"{x}:0N", MAIN_EXCLUDE), "-") + self._add_znak(
            self.parent_rule.equals, "+")
        self._lst_key, self._lst_value = zip(*return_list(rule_text,
                                                          START_FOLDER_NAME.joinpath(self.parent_rule.folders)))
        self._rules_compile_new = re.compile("(" + "$)|(".join(
            replace_template({"+": "[+]", "*": r".*", "!": "[!]", "@": "[@]", "?": "."}, self._lst_key)) + "$)")
        return self

    @staticmethod
    def _get_item(elem, last, index=0):
        return elem[last.lastindex - 1][index]

    def _match_return(self, elem: str):
        znak = ["-", "@", "+", "!"]
        for z in znak:
            ret = self._rules_compile_new.match(z + elem)
            if ret is None or len(ret.string) == 0:
                continue
            keys = self._lst_key[ret.lastindex - 1]
            values = self._lst_value[ret.lastindex - 1]
            yield keys[0], values

    def sort(self, elem: pathlib.Path):
        if elem.is_file():
            c_date = return_time_file(elem, MODIFICATION_TIME)
            self.max_time = max([self.max_time, c_date])
            return CURRENT_DATE - c_date
        else:
            return -1

    @staticmethod
    def reduce(elem: pathlib.Path, files, folders):
        a = {True: (1, 0), False: (0, 1)}[elem.is_file()]
        return map(sum, zip(a, (files, folders)))

    def get_bool_match(self, obj: Analyze):
        """

        :return: exclude, plus, max_files, fast_folder
        """
        exclude = delta("-", obj)
        plus = delta("+", obj)
        fast_folder = delta("@", obj)
        no_max = not delta("!", obj, self.max_time)
        return {"-": exclude, "+": plus, "!": no_max, "@": fast_folder}

    def get_info(self, files: pathlib.Path):
        obj = Analyze(files)
        obj.equals = dict(self._match_return(obj.folders.name))
        obj.lock = any([x[1] for x in obj.equals.values()])
        obj.rule = self.get_bool_match(obj)
        obj.count = get_count(obj.folders)
        return obj

    @property
    def iterdir(self):
        _log = []
        move_old_dir = MOVE_OLD.joinpath(self.parent_rule.folders.as_posix().replace(
            f"{START_FOLDER_NAME.as_posix()}/", ""))

        for files in sorted(self.parent_rule.folders.iterdir(), key=self.sort):
            rules = self.get_info(files)
            if rules.equals and rules.rule["-"] is False:
                rules.deep = self.parent_rule.deep
                if rules.folders.is_dir() and rules.rule["@"] is False:
                    yield rules

                if deleter := Deleter(rules, move_old_dir, self.count):
                    txt, self.count = deleter
                    _log.append(txt)
                continue

            self.count.exclude_files, self.count.exclude_folders = self.reduce(
                rules.folders, self.count.exclude_files, self.count.exclude_folders)

        write_log([f"{' Поиск в: ' + f'{self.parent_rule.folders.as_posix()!r}' + ' ':-^100}", *_log, self.count])


def recursive_dir(dir_name):
    count = Counter()
    with FStat(dir_name) as rules:
        elem: Analyze
        for elem in rules.iterdir:
            count += recursive_dir(elem)
        count += rules.count

    return count


if __name__ == '__main__':
    # Todo: Требуется окончательная проверка.

    # START_FOLDER_NAME, folder, NAME_RULE, NAME_MOVE_DIR, \
    #     NAME_LOG, append_log, size_log, console_out = parse_arg()
    STR_NOW_DATE = datetime.datetime.fromtimestamp(CURRENT_DATE).strftime("%d-%m-%Y")
    argument, logger = alternative_parsers()
    # try:
    #     if NAME_LOG.exists():
    #         log()
    # except AttributeError:
    #     pass
    # tm = datetime.datetime.fromtimestamp(CURRENT_DATE)
    # write_log(f"Current platform: {sys.platform}")
    # write_log(f"{' Начато в: ' + tm.strftime('%d-%m-%Y %H:%M') + ' ':+^100}")
    # MOVE_MAIN_OLD = START_FOLDER_NAME.joinpath(NAME_MOVE_DIR)
    # old_default_rule(MOVE_MAIN_OLD)
    # MOVE_OLD = MOVE_MAIN_OLD.joinpath(STR_NOW_DATE)
    # total_count = Counter()
    # total_parts = Counter()
    #
    # for l_p in folder:
    #     fullpath = START_FOLDER_NAME.joinpath(l_p)
    #     if l_p.exists():
    #         IS_OLD = l_p.name == MOVE_MAIN_OLD.name
    #         total_parts = recursive_dir(l_p)
    #         write_log(f"{' Итог: [' + f'{fullpath.as_posix()!r}' + '] ':*^100}\n{total_parts}")
    #         total_count += total_parts
    #     else:
    #         write_log(f"{fullpath.as_posix()!r} заданная папка не найдена")
    #
    # write_log(f"{'#'*100}\n{' Всего: ':-^100}\n{total_count}")
    # tm_stop = datetime.datetime.now()
    # write_log(f"{' Закончено в: ' + tm_stop.strftime('%d-%m-%Y %H:%M') + ' ':+^100}")
    # tz = datetime.timezone(datetime.timedelta(hours=0))
    # write_log(datetime.datetime.fromtimestamp((tm_stop - tm).total_seconds(),
    #                                           tz=tz).strftime("%H: час., %M: мин., %S: сек., %f: микросек."))
