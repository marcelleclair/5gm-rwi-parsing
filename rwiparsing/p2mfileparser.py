import re
import os
import collections

import numpy as np

# column names for each type of Wireless InSite p2m file
# TODO: path-type files have multiple layers of names, include and identify each layer
headers = {
    # Standard WI Outputs
    "cef": ["path", "E_phi_mag", "E_phi_phs", "E_theta_mag", "E_theta_phs", "E_x_mag", "E_x_phs", "E_y_mag", "E_y_phs", "E_z_mag", "E_z_phs"],
    "cir": ["path", "phase", "mtoa", "power"],
    "doa": ["path", "phi", "theta", "power"],
    "dod": ["path", "phi", "theta", "power"],
    "doppler": ["path", "doppler", "power"],
    "erms": ["rx", "x", "y", "z", "distance", "E_rms"],
    "exmag": ["rx", "x", "y", "z", "distance", "mag", "real", "imaginary"],
    "exphs": ["rx", "x", "y", "z", "distance", "phase", "real", "imaginary"],
    "eymag": ["rx", "x", "y", "z", "distance", "mag", "real", "imaginary"],
    "eyphs": ["rx", "x", "y", "z", "distance", "phase", "real", "imaginary"],
    "ezmag": ["rx", "x", "y", "z", "distance", "mag", "real", "imaginary"],
    "ezphs": ["rx", "x", "y", "z", "distance", "phase", "real", "imaginary"],
    "fspl": ["rx", "x", "y", "z", "distance", "fspl"],
    "fspl0": ["rx", "x", "y", "z", "distance", "fspl0"],  # free space path loss without antenna pattern
    "fspower": ["rx", "x", "y", "z", "distance", "fspower"],  # free space received power
    "fspower0": ["rx", "x", "y", "z", "distance", "fspower0"],  # free space received power without antenna pattern
    "mdoa": ["rx", "x", "y", "z", "distance", "phi", "theta"],
    "mdod": ["rx", "x", "y", "z", "distance", "phi", "theta"],
    "mtoa": ["rx", "x", "y", "z", "distance", "mtoa"],  # mean time of arrival
    "paths": [],  # TODO: this one is tricky, come back to it
    "pg": ["rx", "x", "y", "z", "distance", "pg"],
    "pl": ["rx", "x", "y", "z", "distance", "pl"],
    "power": ["rx", "x", "y", "z", "distance", "power", "phase"],
    "spread": ["rx", "x", "y", "z", "distance", "delayspread"],
    "toa": ["path", "toa", "power"],
    "txloss": ["rx", "x", "y", "z", "distance", "txloss"],
    "xpl": ["rx", "x", "y", "z", "distance", "xpl"],  # excess path loss
    "xpl0": ["rx", "x", "y", "z", "distance", "xpl0"],
    # Comm. Systems Outputs
    "noise": ["rx", "x", "y", "z", "distance", "interference", "noise", "SNR", "SIR", "SINR"],
    "rsum": ["rx", "x", "y", "z", "distance", "strongest_power", "total_power", "total_power_with_phase", "best_SINR", "RSSI"],
    "tp2": ["rx", "x", "y", "z", "distance", "throughput", "capacity", "scheme"]
}
# column formats for each type of Wireless InSite p2m file
# TODO: path-type files have multiple layers of names, include and identify each layer
formats = {
    "cef": [int, float, float, float, float, float, float, float, float, float, float],
    "cir": [int, float, float, float],
    "doa": [int, float, float, float],
    "dod": [int, float, float, float],
    "doppler": [int, float, float],
    "erms": [int, float, float, float, float, float],
    "exmag": [int, float, float, float, float, float, float, float],
    "exphs": [int, float, float, float, float, float, float, float],
    "eymag": [int, float, float, float, float, float, float, float],
    "eyphs": [int, float, float, float, float, float, float, float],
    "ezmag": [int, float, float, float, float, float, float, float],
    "ezphs": [int, float, float, float, float, float, float, float],
    "fspl": [int, float, float, float, float, float],
    "fspl0": [int, float, float, float, float, float],  # free space path loss without antenna pattern
    "fspower": [int, float, float, float, float, float],  # free space received power
    "fspower0": [int, float, float, float, float, float],  # free space received power without antenna pattern
    "mdoa": [int, float, float, float, float, float, float],
    "mdod": [int, float, float, float, float, float, float],
    "mtoa": [int, float, float, float, float, float],  # mean time of arrival
    "paths": [],  # TODO: this one is tricky, come back to it
    "pg": [int, float, float, float, float, float],
    "pl": [int, float, float, float, float, float],
    "power": [int, float, float, float, float, float, float],
    "spread": [int, float, float, float, float, float],
    "toa": [int, float, float],
    "txloss": [int, float, float, float, float, float],
    "xpl": [int, float, float, float, float, float],  # excess path loss
    "xpl0": [int, float, float, float, float, float],
    # Comm. Systems Outputs
    "noise": [int, float, float, float, float, float, float, float, float, float],
    "rsum": [int, float, float, float, float, float, float, float, float, float],
    "tp2": [int, float, float, float, float, float, float, str]
}


class ParsingError(Exception):
    pass


class P2mFileParser:
    """Parser for p2m files. It currently support doa, paths and cir. Notice the regular expression in the code."""

    # project.type.tx_y.rz.p2m
    _filename_match_re = (r'^(?P<project>.*)' +
                          r'\.' +
                          r'(?P<type>\w*)' +
                          r'\.' +
                          r't(?P<transmitter>\d+)'+
                          r'_' +
                          r'(?P<transmitter_set>\d+)' +
                          r'\.' +
                          r'r(?P<receiver_set>\d+)' +
                          r'\.' +
                          r'p2m$')

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self._parse()

    def get_data_dict(self):
        return self.data

    def write_data_dict(self, filename):
        file = open(filename, 'w')
        # TODO: Write header text with variable names, might need to create a new hardcoded list of units
        self._write_dict(self.data, file)
        file.close()

    def _write_dict(self, d, file):
        for key, value in enumerate(d):
            line = ""
            if not isinstance(d, collections.OrderedDict):
                line += str(value) + ' '
            else:
                self._write_dict(value, file)
        line = line.strip() + "\n"
        file.write(line)



    def _parse_meta(self):
        match = re.match(P2mFileParser._filename_match_re,
                         os.path.basename(self.filename))

        self.project = match.group('project')
        self.p2m_type = match.group('type')
        self.transmitter_set = int(match.group('transmitter_set'))
        self.transmitter = int(match.group('transmitter'))
        self.receiver_set = int(match.group('receiver_set'))

    def _parse(self):
        with open(self.filename) as self.file:
            self._parse_meta()
            self.data = collections.OrderedDict()
            while True:
                # read lines until none remain
                try:
                    self._parse_receiver()
                except ParsingError:
                    break
            self.n_receivers = len(self.data)
            self.data["n_receivers"] = self.n_receivers
            self.data.move_to_end("n_receivers", last=False)

    def _parse_receiver(self):
        #raise NotImplementedError()
        line = self._get_next_line()
        sp_line = line.split()
        receiver = int(sp_line[0])
        self.data[receiver] = collections.OrderedDict()
        for index, name in enumerate(headers[self.p2m_type]):
            if index == 0:
                continue
            self.data[receiver][name] = formats[self.p2m_type][index](sp_line[index])  # cast to correct type

    def _get_next_line(self):
        """Get the next uncommedted line of the file

        Call this only if a new line is expected
        """
        if self.file is None:
            raise ParsingError('File is closed')
        while True:
            next_line = self.file.readline()
            if next_line == '':
                raise ParsingError('Unexpected end of file')
            if re.search(r'^\s*#', next_line, re.DOTALL):
                continue
            else:
                return next_line


class P2mPathParser(P2mFileParser):
    """Parser for p2m files containing per-path information (e.g. cef, doa, toa)"""
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self._parse()

    def _parse(self):
        with open(self.filename) as self.file:
            self._parse_meta()
            self._parse_header()
            self.data = collections.OrderedDict()
            self.data["n_receivers"] = self.n_receivers
            for rec in range(self.n_receivers):
                self._parse_receiver()

    def _parse_header(self):
        """read the first line of the file, indicating the number of receivers"""
        line = self._get_next_line()
        self.n_receivers = int(line.strip())

    def _parse_receiver(self):
        # OrderedDict indices start at 1, mirroring the numbering scheme used in WI p2m files
        line = self._get_next_line()
        receiver, n_paths = [int(i) for i in line.split()]
        self.data[receiver] = collections.OrderedDict()
        self.data[receiver]["n_paths"] = n_paths
        for i in range(n_paths):
            line = self._get_next_line()
            sp_line = line.split()
            path = int(sp_line[0])
            self.data[receiver][path] = collections.OrderedDict()
            # self.data[receiver][i] = collections.OrderedDict()
            for index, name in enumerate(headers[self.p2m_type]):
                if index == 0:
                    continue
                self.data[receiver][path][name] = formats[self.p2m_type][index](sp_line[index])  # cast to correct type


if __name__ == '__main__':
    fname = "../example/SA1/WI_ALL_OUTPUTS.power.t001_01.r003.p2m"
    power_p2m = P2mFileParser(fname)
    print("DONE")
    # fname = "../example/iter0.doa.t001_05.r006.p2m"
    # doa = P2mPathParser(fname)
    # data = doa.get_data_dict()
    # phi = []
    # theta = []
    # power = []
    # for i in range(1, data[1]["n_paths"] + 1):
    #     #print(data[1][i])
    #     phi.append(data[1][i]["phi"])
    #     theta.append(data[1][i]["theta"])
    #     power.append(data[1][i]["power"])
    # print("phi = ")
    # print(phi)
    # print("theta = ")
    # print(theta)
    # print("power = ")
    # print(power)
