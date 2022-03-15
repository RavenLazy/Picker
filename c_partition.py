#  coding: utf-8
import shutil
import sys
import re
import datetime
import pathlib
import argparse


class Counter:
    totals = 0
    moves = 0
    delete = 0
    __flag = True

    def __new__(cls, *args, **kwargs):
        if cls.__flag:
            cls.__flag = False
            return super().__new__(Counter)
        return cls


CREATION_TIME = 0
MODIFICATION_TIME = 1
ACCESS_TIME = 2

NAME_MOVE_DIR = "Old"
NAME_RULE = ".rule"
NAME_PATH = ".path"
START_FOLDER_NAME = pathlib.Path.cwd()
CURRENT_DATE = datetime.datetime.now().timestamp()

MAIN_EXCLUDE = [NAME_RULE, NAME_PATH]
DELETE_ALL_OLD = False

DEFAULT_LOG = "log.log"

list_includes_znak = ("@", "!", "+")

compile_rule = re.compile(rf"([-\\{''.join(list_includes_znak)}.*\w\s]+)+[:]*([-\d]+[\w]*)*[:]*([\w])*")
compile_rule1 = re.compile(r"[:]")
old_date_pattern = re.compile(r"\d{2}-\d{2}-\d{4}$")

LOG_TEXT = []


def write_log(text):
    if NAME_LOG:
        lines = type(text) == list

        with open(NAME_LOG, "a") as file:
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
    # Переводим секунды в корректное время
    if elem:
        znak, rt = zip(*[(x, y) for x, y in elem.items() if x in list_includes_znak])
    else:
        rt = None
        znak = "+"

    if rt is None:
        return []
    count = -1
    for seconds, lock in rt:
        count += 1
        if seconds == 0:
            num = 0
            dates = "D:U"
        else:
            num, dates = min(
                [(n, d) for n, d in zip([seconds / 31536000, seconds / 2592000, seconds / 86400, seconds / 3600],
                                        ["Y:U", "M:U", "D:U", "H:U"]) if n - int(n) == 0])
        yield f"{znak[count]}*:{int(num)}{dates}"


def difference_date(date):
    return CURRENT_DATE - date


def get_date(_r):
    return [elem[1] for elem in map(lambda x: x.split(":"), revert_rules(_r)) if elem[0][0] in ("+", "!")]


def replace_template(pat, item):
    items = item
    for elem in items:
        for key, value in pat.items():
            elem = elem.replace(key, value)
        yield elem


def normal_date(item: str):
    codes: str
    times: str | int

    times, codes = re.findall(r"([-\d]+)+(\w)", item)[0]

    times = int(times)

    ret = {"d": datetime.timedelta(days=1 * times).total_seconds(),
           "m": datetime.timedelta(days=30 * times).total_seconds(),
           "y": datetime.timedelta(days=365 * times).total_seconds(),
           "h": datetime.timedelta(hours=1 * times).total_seconds(),
           "n": CURRENT_DATE,
           "e": 0}
    return ret[codes.lower()]


def decompress(el: list, folders: pathlib.Path):
    count = 0
    for item in el:
        count += 1
        pat = compile_rule1.split(item, maxsplit=2)
        pat += [''] * (3 - len(pat))
        name, dates, lock = pat
        try:
            dates = int(normal_date(dates)) or (0 if name[0] in list_includes_znak else int(CURRENT_DATE))
        except IndexError:
            print(f"{item!r} ошибка разметки в списке правил {NAME_RULE!r}."
                  f"\nПапка: {folders.as_posix()!r}.\nСтрока: {count}")
            sys.exit(1)
        lock = False if lock == "U" else True   # True - папка не будет удалена
        yield name, dates, lock


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

    def _delta(self, znak):
        fl = self.equals.get(znak, False)
        if fl is False:
            return {"-": True, "+": False, "!": False, "@": False}[znak]
        date, _ = fl
        if self.is_max and znak != "@":
            return CURRENT_DATE - self.max_time > date
        return CURRENT_DATE - return_time_file(self.filename, self.type_time) > date

    def exclude(self):
        return not self._delta("-")

    def include(self):
        return self._delta("+") or self._delta("!")

    def is_dir(self):
        if self.filename.is_file():
            return False
        return True

    def fast_del(self):
        # if self.filename.is_file():
        #     return False
        if IS_OLD:
            if self.filename.is_file():
                self.filename.unlink(missing_ok=True)
            else:
                self.filename.rmdir()
        else:
            self.move_to_old()
        return True

    def is_fast(self):
        if self.is_dir():
            fast = self._delta("@")
            print(f"Удалить эту директорию: {self.filename} -> {fast}")
            return fast
        return False

    def move_to_old(self):
        move_files = self.move_dir.joinpath(self.filename.name)
        if not self.move_dir.exists():
            self.move_dir.mkdir(parents=True)
        self.filename.replace(move_files)

    def iterdir(self):
        return self.filename.iterdir()

    def del_dir(self):
        if self.filename.is_dir():
            is_contains = len([x for x in self.filename.iterdir()]) != 0
            _, values = self.equals.get('+') or self.equals.get("!", (0, "U"))
            closed = {"L": True, "U": False}[values]
            if is_contains is False and closed is False:
                self.filename.rmdir()

    def is_old_delete(self):
        if IS_OLD and DELETE_ALL_OLD:
            r = re.search(old_date_pattern, self.filename.as_posix())
            if r and self.include():
                shutil.rmtree(self.filename, ignore_errors=True)
            return True
        return False


class FStat:
    """Обработка путей"""

    def __init__(self, rule, type_time=MODIFICATION_TIME):
        self._lot = Job(rule)
        self._lot.type_time = type_time
        if isinstance(rule, Job):
            self.directory = pathlib.Path(rule.filename)
            self._parent = revert_rules(rule.equals)
            self._lot.is_max = rule.equals.get("!") is not None
        else:
            self.directory = pathlib.Path(rule)
            self._parent = []
        self._lot.move_dir = MOVE_OLD / self.directory.as_posix().replace(f"{START_FOLDER_NAME.as_posix()}/", "")

    def __exit__(self, exc_type, exc_val, exc_tb):
        #  Проверка на пустоту каталога и возможность его удаления
        if exc_val:
            print(exc_type, exc_tb, exc_val)
            return False

    @staticmethod
    def _return_list(text, folders):
        tmp = []
        for name, times, lock in decompress(text, folders):
            if name in tmp:
                continue
            tmp.append(name)
            yield name, (times, lock)

    @staticmethod
    def _add_znak(item, znak):
        return list(map(lambda x: znak + x if x[0] not in list_includes_znak else x, item))

    def __enter__(self):
        # Читаем файл с правилами, если найден, добавляем правила по умолчанию.
        # На выходе получаем список кортежей с дельта-временем, определением можно ли удалить папку и именем этой папки

        rules = self.directory.joinpath(NAME_RULE)
        rule_text = (rules.read_text(encoding="utf-8").splitlines() if rules.exists() else [])
        rule_text += self._add_znak(map(lambda x: f"{x}:0N", MAIN_EXCLUDE), "-") + self._add_znak(self._parent, "+")
        self._lst_key, self._lst_value = zip(*self._return_list(rule_text, START_FOLDER_NAME.joinpath(self.directory)))
        self._rules_compile_new = re.compile("(" + "$)|(".join(
            replace_template({"+": "[+]", "*": r".+", "!": "[+]", "@": "[@]"}, self._lst_key)) + "$)")
        return self

    @staticmethod
    def _get_item(elem, last, index=0):
        return elem[last.lastindex - 1][index]

    def _match_return(self, elem):
        znak = ["-", "@", "+"]
        for z in znak:
            ret = self._rules_compile_new.match(z + elem)
            if ret is None:
                continue
            yield self._get_item(self._lst_key, ret), (
                self._get_item(self._lst_value, ret),
                {True: "L", False: "U"}.get(self._get_item(self._lst_value, ret, 1), True))

    def iterdir(self):
        if self._lot.is_max:
            self._lot.max_time = max([return_time_file(x, self._lot.type_time)
                                      for x in self.directory.iterdir() if x.is_file()] or [CURRENT_DATE])

            # print(datetime.datetime.fromtimestamp(self._lot.max_time))
        for self._lot.filename in sorted(self.directory.iterdir(), key=lambda x: x.is_dir()):
            mm = self._match_return(self._lot.filename.name)
            self._lot.equals = dict(mm)
            yield self._lot


def recursive_dir(rr: pathlib.Path):
    txt = ' Ищу в ' + f"{rr.as_posix()!r} "
    LOG_TEXT.append(f"{txt:-^100}")
    print(f"{txt:-^}")
    #  Todo: Добавить еще один аргумент в файл с правилами для вложенных папок. Если включено elem.is_max,
    #   то вложенные папки обрабатываются по времени самой папки, а не файлов!
    with FStat(rr, type_time=MODIFICATION_TIME) as rules:
        for elem in rules.iterdir():
            if elem.exclude():
                continue
            if elem.is_dir():
                if elem.is_fast():
                    print(f"Delete {elem.filename}")
                    elem.fast_del()
                else:
                    recursive_dir(elem)
                elem.del_dir()
                continue
            if elem.include():
                elem.fast_del()


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
    coeff = 1000
    size, ident = re.findall(r"([\d]+)([MmKkBb]?)", item)[0]
    size = int(size)
    ret = {"m": size * (coeff ** 2), "k": size * coeff, "b": size * 1}.get(ident.lower(), size)
    return ret


def parse_arg():
    arg = argparse.ArgumentParser(description="Перенос старых файлов в отдельную директорию с сохранением путей")
    group1 = arg.add_mutually_exclusive_group()
    group2 = arg.add_mutually_exclusive_group()

    arg.add_argument("--parent", "-p", nargs="*", default=START_FOLDER_NAME, metavar=f"{'Folder'!r}", type=pathlib.Path,
                     help=f"Папка с файлом {NAME_PATH}. "
                          f"По умолчанию папка запуска скрипта {START_FOLDER_NAME.as_posix()!r}")

    group1.add_argument("--pathname", "-pn", type=str, default=NAME_PATH, nargs=1, metavar=".path",
                        help=f"Name of the path file. Default: {NAME_PATH!r}")

    arg.add_argument("--rule", "-r", type=str, default=NAME_RULE, nargs=1, metavar=".rule",
                     help=f"Имя файла с правилами анализа: По умолчанию: {NAME_RULE!r}")

    arg.add_argument("--delete_old", '-d', action="store_true", default=DELETE_ALL_OLD,
                     help=f"Удаляем все старые файлы из папки переноса. Анализируется только основная папка. "
                          f"По умолчанию, анализируется каждый файл.")

    arg.add_argument("--old", "-o", nargs=1, type=str, default=NAME_MOVE_DIR, metavar=f"{'Folder/Old'!r}",
                     help=f"Имя папки в которую будут переносится старые файлы. По умолчанию: {NAME_MOVE_DIR}")

    group1.add_argument("-s", "--search", type=str, nargs="+", action="append",
                        metavar=f"{'Folder/1'!r} {'Folder/2'!r} ...", help="Папки поиска")

    group2.add_argument("-l", "--log", action="store_true",
                        help="Разрешить запись лога. Задается без дополнительного аргумента. По умолчанию лог не пишем")

    group2.add_argument("-lf", "--log_file", action="store", metavar="log file",
                        help="Имя файла для записи лога. Если указано будем писать в этот файл. "
                             "Указывается либо имя файла, либо параметр '-l (--log)'")

    arg.add_argument("-al", "--append_log", action="store_false",
                     help="Начинать писать лог файл заново. По умолчанию дописывать")

    arg.add_argument("-sl", "--size_log", type=for_size, default=1000,
                     help="Размер лог файла в байтах. По умолчанию 1000 байт. "
                          "При превышении файл будет переименован и создан новый")
    arg_s = arg.parse_args()
    sfname: pathlib.Path

    sfname, npath, nrule, dallold, nmovedir, maindir, log_write, name_log, app_log, s_log = return_list_main(arg_s)

    if sfname.exists() is False:
        raise OSError(f"{START_FOLDER_NAME.as_posix()!r} not exists")

    if log_write or name_log:
        name_log = pathlib.Path(name_log or DEFAULT_LOG)

    name_path = maindir or npath
    ss = []
    if maindir:
        ss = [elem[0] if type(elem) == list else elem for elem in name_path]
    else:
        _start = sfname.joinpath(name_path)
        if _start.exists():
            ss = _start.read_text().splitlines()

    ss = [re.sub(r"(%start%)", sfname.as_posix(), elem) for elem in ss]
    ss = map(pathlib.Path, ss)

    return sfname, ss, nrule, dallold, nmovedir, name_log, app_log, s_log


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


if __name__ == '__main__':
    START_FOLDER_NAME, folder, NAME_RULE, DELETE_ALL_OLD, NAME_MOVE_DIR, NAME_LOG, append_log, size_log = parse_arg()
    STR_NOW_DATE = datetime.datetime.fromtimestamp(CURRENT_DATE).strftime("%d-%m-%Y")
    try:
        if NAME_LOG.exists():
            log()
    except AttributeError:
        pass
    write_log(f"{' '.join([' Start write at:', STR_NOW_DATE, ' ']):-^100}")
    write_log(f"Current platform: {sys.platform}")
    MOVE_MAIN_OLD = START_FOLDER_NAME.joinpath(NAME_MOVE_DIR)
    MOVE_OLD = MOVE_MAIN_OLD.joinpath(STR_NOW_DATE)
    print(MOVE_OLD)
    for l_p in folder:
        fullpath = START_FOLDER_NAME.joinpath(l_p)
        if l_p.exists():
            IS_OLD = l_p == MOVE_MAIN_OLD
            # print("#" * 50)
            print(fullpath.as_posix())
            recursive_dir(l_p)
        else:
            write_log(f"{fullpath.as_posix()!r} заданная папка не найдена")
            print(f"{fullpath.as_posix()!r} not exists")
