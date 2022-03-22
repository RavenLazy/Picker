#  coding: utf-8
import sys
import re
import datetime
import pathlib
import argparse

CREATION_TIME = 0
MODIFICATION_TIME = 1
ACCESS_TIME = 2

NAME_MOVE_DIR = "Old"
NAME_RULE = ".rule"
NAME_PATH = ".path"
START_FOLDER_NAME = pathlib.Path.cwd()
a = datetime.datetime.now().strftime("%d-%m-%Y %H")
CURRENT_DATE = datetime.datetime.strptime(a, "%d-%m-%Y %H").timestamp()
del a

MAIN_EXCLUDE = [NAME_RULE, NAME_PATH]
DELETE_ALL_OLD = False

DEFAULT_LOG = "log.log"

list_includes_znak = ("@", "!", "+")

compile_rule = re.compile(rf"([-\\{''.join(list_includes_znak)}.*\w\s]+)+[:]*([-\d]+[\w]*)*[:]*([\w])*")
compile_rule1 = re.compile(r"[:]")
old_date_pattern = re.compile(r"\d{2}-\d{2}-\d{4}$")


# LOG_TEXT = []


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
    return {"m": size * (coeff ** 2), "k": size * coeff, "b": size * 1}.get(ident.lower(), size)


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

    arg.add_argument("--name_old", "-n_o", nargs=1, type=str, default=NAME_MOVE_DIR, metavar=f"{'Folder/Old'!r}",
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

    arg.add_argument("-o", "--out", action="store_false",
                     help="Выводить информацию в консоль? По умолчанию: Да")

    arg_s = arg.parse_args()
    sfname: pathlib.Path

    sfname, npath, nrule, dallold, nmovedir, maindir, log_write, \
    name_log, app_log, s_log, c_out = return_list_main(arg_s)

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

    return sfname, ss, nrule, dallold, nmovedir, name_log, app_log, s_log, c_out


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


def write_log(text):
    lines = type(text) == list
    #  Выводим текст в консоль
    if console_out:
        if lines:
            print(*text, sep="\n")
        else:
            print(text)

    #  Выводим текст в файл
    if NAME_LOG:
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
    # Заданное время в секунды
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
    # Раскладываем время из файла правил
    count = 0
    for item in el:
        count += 1
        pat = compile_rule1.split(item, maxsplit=2)
        pat += [''] * (3 - len(pat))
        name, dates, lock = pat
        dates = dates or '0N'
        try:
            dates = int(normal_date(dates)) or (0 if name[0] in list_includes_znak else int(CURRENT_DATE))
        except IndexError:
            print(f"{item!r} ошибка разметки в списке правил {NAME_RULE!r}."
                  f"\nПапка: {folders.as_posix()!r}.\nСтрока: {count}")
            sys.exit(1)
        lock = False if lock == "U" else True  # True - папка не будет удалена
        yield name, dates, lock


class Counter:
    __slots__ = ("total", "folder", "files", "exclude", "move_object", "delete_files", "delete_folders")

    text = ("Объектов: ", "Количество папок: ", "Учтенных файлов: ", "Пропущенных файлов: ",
            "Перемещено: ", "Удалено файлов: ", "Удалено папок: ")

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
            [f"{key}{value}" for key, value in zip(self.text, [self.__getattribute__(x) for x in self.__slots__])])


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
            return {"-": True, "+": False, "!": False, "@": False}[znak]
        date, _ = fl
        if self.is_max and znak != "@":
            return CURRENT_DATE - self.max_time > date
        return CURRENT_DATE - return_time_file(self.filename, self.type_time) > date

    def exclude(self):
        ret = not self._delta("-")
        if ret:
            self.counter.exclude += 1
        return ret

    def include(self):
        return self._delta("+") or self._delta("!")

    def is_dir(self):
        if self.filename.is_file():
            self.counter.files += 1
            return False
        return True

    def fast_del(self):
        # if IS_OLD:
        #     self.log.append(f"{self.filename.as_posix()!r} Время хранения истекло - удаляем!")
        #     if self.filename.is_file():
        #         self.counter.delete_files += self.del_or_move(self.filename.unlink, missing_ok=True)
        #     else:
        #         self.counter.delete_folders += self.del_or_move(self.filename.rmdir)
        # else:
        self._move_to_old()
        return True

    def is_fast(self):
        if self.is_dir():
            self.counter.folder += 1
            fast = self._delta("@")
            return fast
        return False

    def _move_to_old(self):
        move_files = self.move_dir.joinpath(self.filename.name)
        self.log.append(f"Время {fullpath.joinpath(self.filename.name).as_posix()!r} вышло за заданный диапазон. "
                        f"Переносим в {move_files.as_posix()!r}")
        if not self.move_dir.exists():
            self.move_dir.mkdir(parents=True)
        self.counter.move_object += self.del_or_move(self.filename.replace, move_files)


class JobOld(Job):

    def fast_del(self):
        self.log.append(f"{self.filename.as_posix()!r} Время хранения истекло - удаляем!")
        if self.filename.is_file():
            self.counter.delete_files += self.del_or_move(self.filename.unlink, missing_ok=True)
        else:
            self.counter.delete_folders += self.del_or_move(self.filename.rmdir)


class FStat:
    """Обработка путей"""

    def __init__(self, rule: Job | str, type_time=MODIFICATION_TIME):
        if IS_OLD:
            self._lot = JobOld(rule)
        else:
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
        global NAME_LOG
        if exc_val:
            NAME_LOG = NAME_LOG or "error.log"
            write_log([exc_type, exc_tb, exc_val])
            return False
        txt = ' Ищу в: ' + f"{self.directory.as_posix()!r} "
        write_log(f"{txt:-^100}")
        if self._lot.log:
            write_log(self._lot.log)
            write_log('='*100)
        write_log(self.get_counter())

    def get_counter(self):
        return self._lot.counter

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

        for self._lot.filename in sorted(self.directory.iterdir(), key=lambda x: x.is_file()):
            self._lot.counter.total += 1
            mm = self._match_return(self._lot.filename.name)
            self._lot.equals = dict(mm)
            yield self._lot


def recursive_dir(rr):
    global total_count
    #  Todo: Добавить еще один аргумент в файл с правилами для вложенных папок. Если включено elem.is_max,
    #   то вложенные папки обрабатываются по времени самой папки, а не файлов!

    count = Counter()
    with FStat(rr) as rules:
        elem: Job
        for elem in rules.iterdir():
            if elem.exclude():
                continue
            if elem.is_dir():
                if elem.is_fast():
                    elem.fast_del()
                else:
                    count += recursive_dir(elem)
                continue
            if elem.include():
                elem.fast_del()
        count += rules.get_counter()

    return count


if __name__ == '__main__':
    START_FOLDER_NAME, folder, NAME_RULE, DELETE_ALL_OLD, NAME_MOVE_DIR, \
    NAME_LOG, append_log, size_log, console_out = parse_arg()
    STR_NOW_DATE = datetime.datetime.fromtimestamp(CURRENT_DATE).strftime("%d-%m-%Y")
    try:
        if NAME_LOG.exists():
            log()
    except AttributeError:
        pass
    write_log(f"{' '.join([' Start write at:', datetime.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ']):-^100}")
    write_log(f"Current platform: {sys.platform}")
    MOVE_MAIN_OLD = START_FOLDER_NAME.joinpath(NAME_MOVE_DIR)
    MOVE_OLD = MOVE_MAIN_OLD.joinpath(STR_NOW_DATE)
    total_count = Counter()
    total_parts = Counter()

    for l_p in folder:
        fullpath = START_FOLDER_NAME.joinpath(l_p)
        if l_p.exists():
            IS_OLD = l_p == MOVE_MAIN_OLD
            total_parts = recursive_dir(l_p)
            s = ' '.join(['*', '[\'' + fullpath.as_posix() + '\']', '*'])
            write_log(f"{s:*^100}\n{total_parts}")
            total_count += total_parts
        else:
            write_log(f"{fullpath.as_posix()!r} заданная папка не найдена")

    write_log(f"{'#' * 100}\nВсего:\n{'_' * 100}\n{total_count}")
