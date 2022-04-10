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
NAME_LOG = "log.log"

CURRENT_DATE = datetime.datetime.now().replace(second=0, microsecond=0).timestamp()

MAIN_EXCLUDE = [NAME_RULE, NAME_PATH]

list_includes_znak = ("@", "!", "+")

old_date_pattern = re.compile(r"\d{2}-\d{2}-\d{4}$")

counter_text = ("Объектов", "Количество папок", "Количество файлов", "Пропущено файлов", "Пропущено папок",
                "Перемещено", "Удалено файлов", "Удалено папок")


def add_default_rule():
    if not arguments.trash.exists():
        arguments.trash.mkdir(parents=True)
    if not (rule := arguments.trash.joinpath(arguments.rule)).exists():
        with rule.open("w") as f:
            f.writelines(Default_rule_old_days)
    return None


def add_in_list_set(array: list, text: str):
    """Создаем список с неповторяющимися элементами"""
    if text not in array:
        return array + [text]
    return array


def decompress(el: list, folders: pathlib.Path, dp: bool):
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
                write_log(f"{item!r} ошибка разметки в списке правил {NAME_RULE!r}."
                          f"\nПапка: {folders.as_posix()!r}.\nСтрока: {count}")
                sys.exit(1)
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
    for v in values:
        r = folder.joinpath(v)
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
        setattr(namespace, 'Search', read_path(namespace, values))


class ActionSearch(ActionFile):

    def __call__(self, parser, namespace, values: list, option_string=None):
        values = create_path(namespace, [values])
        if atr := getattr(namespace, 'Search'):
            if type(atr) == list:
                _, values = atr.extend(values), atr
        setattr(namespace, self.dest, list(values))


class ActionTrash(ActionFile):

    def __call__(self, parser, namespace, values, option_string=None):
        values = list(create_path(namespace, values))[0]
        setattr(namespace, self.dest, values)


def rechange(args: argparse.ArgumentParser, line, value):
    ret = args.parse_args([line.command, line.folder.as_posix(), *value])
    return ret


def alternative_parsers():

    arg = argparse.ArgumentParser(description="Перенос старых файлов в отдельную директорию с сохранением путей",
                                  prog="Trash")

    search_folder = arg.add_argument_group()
    logged = arg.add_argument_group()

    arg.add_argument("command", choices=["execute", "test"],
                     help="<execute> работа по заданным правилам. <test> прогон по папкам без изменения.")

    arg.add_argument("folder", metavar="<start folder name>", action="store",
                     type=pathlib.Path, help=f"Откуда начинаем искать")

    search_folder.add_argument("-Search", type=str, action=ActionSearch,
                               metavar="<Folder1> <Folder2> ... <FolderNNN>",
                               help=f"Папки поиска.")
    search_folder.add_argument("-pathname", type=str, default='.path', metavar=".path", action=ActionFile,
                               help=f"Имя файла с папками для поиска. По умолчанию: <%(default)s>")

    arg.add_argument("-rule", default='.rule', metavar="<'.rule'>",
                     help="Имя файла с правилами. По умолчанию: %(metavar)s")

    arg.add_argument("-trash", nargs=1, metavar="<'Trash'>", type=str, action=ActionTrash,
                     help=f"Имя папки в которую будут переносится старые файлы. По умолчанию: %(metavar)s")

    logged.add_argument("-log", action="store_true",
                        help="Включает запись лога. Дополнительные параметры [-name log.log] [-size 1000] [-append]"
                        "По умолчанию: <%(default)s>")

    arg.add_argument("-console", action="store_true", help="Выводить в консоль. По умолчанию <%(default)s>")

    arg.add_argument('--version', "-V", action='version', version='%(prog)s 1.0')

    # Выставляем параметры для лог файла, если указан аргумент -log
    logging: bool
    if logging := arg.parse_known_args()[0].log:
        logged.add_argument("-name", nargs=1, action=ActionTrash, help=f"Имя лог-файла <{NAME_LOG}>")
        logged.add_argument("-size", default=1000, type=for_size, help="Размер лога <1000>")
        logged.add_argument("-append", action="store_true", help="Продолжать дописывать <True>")

    command_argument = arg.parse_args()
    # Имя лог файла
    if logging and command_argument.name is None:
        arg.set_defaults(name=rechange(arg, command_argument, ['-n', NAME_LOG]).name)

    # Добавляем имя файла с ошибками
    arg.set_defaults(error_log=command_argument.folder.joinpath(ERROR_LOG))

    # Создаем список папок для поиска. Читаем файл .path или что задано в pathname
    if command_argument.Search is None:
        arg.set_defaults(Search=list(
            *rechange(arg, command_argument, ['-p', '.path']).Search
        ) or [command_argument.folder])

    if command_argument.trash is None:
        arg.set_defaults(trash=rechange(arg, command_argument, ['-t', NAME_MOVE_DIR]).trash)

    arg.set_defaults(trash_day=(arg.get_default('trash') or command_argument.trash).joinpath(STR_NOW_DATE))
    arg.set_defaults(Search=arg.get_default("Search") + [arg.get_default("trash") or command_argument.trash])
    arg.set_defaults(work=command_argument.command == 'execute')

    return arg.parse_args()


def log():
    if arguments.log and arguments.name.exists():
        if arguments.name.stat().st_size >= arguments.size:
            name = arguments.name.as_posix().split(".")[0]
            fullname = pathlib.Path('-'.join([name, STR_NOW_DATE + arguments.name.suffix]))
            count = 0
            while fullname.exists():
                count += 1
                fullname = pathlib.Path(
                    '-'.join([name, STR_NOW_DATE + f"_{str(count).zfill(3)}{arguments.name.suffix}"]))
            arguments.name.replace(fullname)


def write_log(text, err_log=False):
    name_log = arguments.error_log if err_log else arguments.name
    if len(text) > 0:
        lines = type(text) == list
        #  Выводим текст в консоль
        if arguments.console:
            if lines:
                print(*text, sep="\n")
            else:
                print(text)

        #  Выводим текст в файл
        if arguments.log or err_log:
            with open(name_log, "a") as f:
                if lines:
                    f.writelines([f"{line}\n" for line in text])
                else:
                    f.write(text + '\n')


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
            if arguments.is_old:
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
    if arguments.work:
        shutil.rmtree(elem.folders.as_posix(), ignore_errors=True)


def delete(elem: Analyze):
    """Удаляем файл"""
    if arguments.work:
        elem.folders.unlink(missing_ok=True)


def replace(elem: Analyze, old_dir: pathlib.Path):
    """Перемещаем файл"""
    if arguments.work:
        if old_dir.exists() is False:
            old_dir.mkdir(parents=True)
        elem.folders.replace(old_dir)


class FStat:
    """Обработка путей"""

    def __init__(self, rule: Analyze | str):

        self.max_time = 0
        self.parent_rule = Analyze(rule)
        self.count = self.parent_rule.count

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            write_log([exc_type, exc_tb, exc_val], err_log=True)
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
                                                          arguments.folder.joinpath(self.parent_rule.folders)))
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

    def get_bool_match(self, obj: Analyze) -> dict:
        exclude = delta("-", obj)
        plus = delta("+", obj)
        fast_folder = delta("@", obj)
        no_max = not delta("!", obj, self.max_time)
        return {"-": exclude, "+": plus, "!": no_max, "@": fast_folder}

    def get_info(self, files: pathlib.Path) -> Analyze:
        obj = Analyze(files)
        obj.equals = dict(self._match_return(obj.folders.name))
        obj.lock = any([x[1] for x in obj.equals.values()])
        obj.rule = self.get_bool_match(obj)
        obj.count = get_count(obj.folders)
        return obj

    @property
    def iterdir(self):
        _log = []
        move_old_dir = arguments.trash_day.joinpath(self.parent_rule.folders.as_posix().replace(
            f"{arguments.folder.as_posix()}/", ""))
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
    STR_NOW_DATE = datetime.datetime.fromtimestamp(CURRENT_DATE).strftime("%d-%m-%Y")
    arguments: argparse.Namespace
    arguments = alternative_parsers()
    print(arguments, sep='\n')
    log()
    write_log(f"Current platform: {sys.platform}")
    tm = datetime.datetime.fromtimestamp(CURRENT_DATE)
    write_log(f"{' Начато в: ' + tm.strftime('%d-%m-%Y %H:%M') + ' ':+^100}")
    write_log(work := f'Запуск осуществлен с аргументом {arguments.command!r}. '
                      f'Файлы {"обрабатываются" if arguments.work else "не обрабатываются"}!')
    total_count = Counter()
    total_parts = Counter()
    for file in arguments.Search:
        arguments.is_old = file == arguments.trash
        if arguments.is_old:
            add_default_rule()
        if file.exists():
            total_parts = recursive_dir(file)
            write_log(f"{' Итог: [' + f'{arguments.folder.as_posix()!r}' + '] ':*^100}\n{total_parts}")
            total_count += total_parts
        else:
            write_log(f"{arguments.folder.as_posix()!r} заданная папка не найдена")

    write_log(f"{'#'*100}\n{' Всего: ':-^100}\n{total_count}")
    tm_stop = datetime.datetime.now()
    write_log(f"{' Закончено в: ' + tm_stop.strftime('%d-%m-%Y %H:%M') + ' ':+^100}")
    tz = datetime.timezone(datetime.timedelta(hours=0))
    write_log(datetime.datetime.fromtimestamp((tm_stop - tm).total_seconds(),
                                              tz=tz).strftime("%H: час., %M: мин., %S: сек., %f: микросек."))
    write_log(work)
