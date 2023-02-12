#!/user/bin/env python3
# coding: utf-8

import os
import argparse
from logging import getLogger, StreamHandler, Formatter, INFO, DEBUG
import csv
import datetime
from decimal import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# tool version
version = 1.0

# logger settings
logger = getLogger(__name__)
handler = StreamHandler()
formatter = Formatter('%(asctime)s %(levelname)-7s %(message)s', '%Y/%m/%d %H:%M:%S')
handler.setFormatter(formatter)
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)

# Custome Exception
class MyException(Exception):
    def __init__(self, msg=''):
        super().__init__()
        self.message = msg

# For CSV Control
class CsvControl(object):
    def __init__(self, file, logger=None, debug=False):
        self.logger = logger or getLogger(__name__)
        self.debug = debug
        self.file = file
        self.csv_lines = []  # original data
        self.csv_data = [] # filtered data

    def _read_csv(self):
        logger.info('Read csv file: {}'.format(self.file))
        with open(self.file, 'r', encoding='utf-8-sig') as f: # BOMなしUTF-8
            lines = csv.reader(f)
            for line in lines:
                self.csv_lines.append(line)

        if self.debug:
            for line in self.csv_lines:
                print(line)

        logger.info('Successfully to read csv file')

    def read_csv(self):
        if not self.csv_lines:
            self._read_csv()

        self.csv_data = self.csv_lines.copy()
        if self.debug:
            for line in self.csv_data:
                print(line)

    def aggregate_data(self):
        pass

# For Lion FX
class LionFX(CsvControl):
    def __init__(self, file, logger=None, debug=False):
        super().__init__(file,logger,debug)
        self.all_data_lines = []
        self.target_lines = []

    def read_csv(self):
        if not self.csv_lines:
            self._read_csv()

        logger.info('Filtered lines')
        self.csv_data = list(filter(lambda line:line[0], self.csv_lines))
        del self.csv_data[0]
        if self.debug:
            for line in self.csv_data:
                print(line)

    # Set entory kind
    def _get_entry_kind(self,kind):
        return 'short' if kind == '買' else 'long'

    def _aggregate_data(self):
        if not self.csv_data:
            self.read_csv()

        logger.info('Aggregate csv data for Lion FX')
        for line in self.csv_data:
            self.all_data_lines.append({'start_time':line[11],'end_time':line[0],
                                      'kind':self._get_entry_kind(line[9]),
                                      'start_point':line[12],'end_point':line[13],
                                      'pips':line[14],'id':line[2]})
#        self.all_data_lines = sorted(data_lines, key=lambda x:x['start_time'])
        if self.debug:
            for row in self.all_data_lines:
                print(row)

        logger.info('Successfully aggregate csv data')

    def _replace_time_unit(self,time):
        return time.replace('m','').replace('h','')

    def get_timedate(self, time_str):
        return datetime.datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')

    def _get_time_unit(self, time, segment):
        dt = self.get_timedate(time)
        interval = int(self._replace_time_unit(segment))
        if 'h' in segment:
            _hour = int(dt.hour/interval)*interval
            _dt_mod = datetime.datetime(dt.year,dt.month,dt.day,_hour)
            if self.debug:
                print('{} -> {}'.format(dt.hour,_hour))
        else:
            _minute = int(dt.minute/interval)*interval
            _dt_mod = datetime.datetime(dt.year,dt.month,dt.day,dt.hour,_minute)
            if self.debug:
                print('{} -> {}'.format(dt.minute,_minute))
        return _dt_mod.strftime('%Y/%m/%d %H:%M:%S')

    def _filter_aggregate_data(self, from_time, to_time):
        if not self.all_data_lines:
            self._aggregate_data()

        logger.info('Aggregation range: {} - {}'.format(from_time,to_time))

        _target_lines = list(filter(lambda _dict: _dict['start_time'] > from_time, self.all_data_lines))
        self.target_lines = list(filter(lambda _dict: _dict['end_time'] < to_time, _target_lines))

        if not self.target_lines:
            raise MyException('Aggregation data is not found')

    # not use
    def _get_aggregate_data_(self, from_time, to_time, segment='5m'):
        if not self.target_lines:
            self._filter_aggregate_data(from_time, to_time)

        logger.info('Sort by time')
        result_dict = {}
        for line in self.target_lines:
            for mode in ['start','end']:
                time = line['{}_time'.format(mode)]
                _ti = self._get_time_unit(time,segment)
                if not _ti in result_dict.keys():
                    result_dict[_ti] = []
                division = 'in' if mode == 'start' else 'out'
                result_dict[_ti].append({'id':line['id'],'point':line['{}_point'.format(mode)],
                                         'division':division})
        if self.debug:
            for key,value in result_dict.items():
                print(key)
                print(value)

        logger.info('Successfully to sort by time')
        return result_dict

    def get_aggregate_data(self, from_time, to_time, segment='5m'):
        result_list = []
        if not self.target_lines:
            self._filter_aggregate_data(from_time, to_time)

        logger.info('Filter by trade set')
        for d in self.target_lines:
            s_time = self._get_time_unit(d['start_time'],segment)
            e_time = self._get_time_unit(d['end_time'],segment)
            result_list.append({
                'start_time' : self.get_timedate(s_time),
                'end_time'   : self.get_timedate(e_time),
                'start_point': float(d['start_point']),
                'end_point'   : float(d['end_point']),
                'kind':d['kind']
            })
            logger.info('{} - {} / {} - {} / {}'.format(
                        s_time,e_time,d['start_point'],d['end_point'],d['kind']))

        logger.info('The count of trade sets is {}'.format(len(result_list)))
        return result_list

    def _get_margin_time(self, time, segment, is_next=False):
        dt = self.get_timedate(self._get_time_unit(time,segment))
        interval = int(self._replace_time_unit(segment))
        if 'h' in segment:
            dt = dt + datetime.timedelta(hours=interval) \
                    if is_next else dt - datetime.timedelta(hours=interval)
        else:
            dt = dt + datetime.timedelta(minutes=interval) \
                    if is_next else dt - datetime.timedelta(minutes=interval)
        return dt.strftime('%Y/%m/%d %H:%M:%S')

    # not use
    def _get_range_base_(self, target_lines, segment='5m'):
        _start = min(min(_dict['start_time'] for _dict in target_lines),
                     min(_dict['end_time'] for _dict in target_lines))
        _end = max(max(_dict['start_time'] for _dict in target_lines),
                   max(_dict['end_time'] for _dict in target_lines))
        _max = max(max(_dict['start_point'] for _dict in target_lines),
                   max(_dict['end_point'] for _dict in target_lines))
        _min = min(min(_dict['start_point'] for _dict in target_lines),
                   min(_dict['end_point'] for _dict in target_lines))

        start_time = self._get_margin_time(_start, segment)
        end_time = self._get_margin_time(_end, segment, is_next=True)

        max_point = float(Decimal(_max).quantize(Decimal('.1'),rounding=ROUND_UP))
        min_point = float(Decimal(_min).quantize(Decimal('.1'),rounding=ROUND_DOWN))
#        max_point = '{:.3f}'.format(Decimal(_max).quantize(Decimal('.1'),rounding=ROUND_UP) + Decimal('0.1'))
#        min_point = '{:.3f}'.format(Decimal(_min).quantize(Decimal('.1'),rounding=ROUND_DOWN) - Decimal('0.1'))

        interval = self._replace_time_unit(segment) if 'm' in segment \
                   else int(self._replace_time_unit(segment)) * 60

        if self.debug:
            print('{} - {} / {} - {}'.format(_start,_end,_min,_max))
            print('{} - {} / {} - {}'.format(start_time,end_time,min_point,max_point))
            print(interval)

        return {'start':start_time, 'end':end_time,
                'max':max_point, 'min':min_point,
                'segment':interval}

    def _get_range_base(self, target_lines):
        _max = max(max(_dict['start_point'] for _dict in target_lines),
                   max(_dict['end_point'] for _dict in target_lines))
        _min = min(min(_dict['start_point'] for _dict in target_lines),
                   min(_dict['end_point'] for _dict in target_lines))

        max_point = float(Decimal(_max).quantize(Decimal('.1'),rounding=ROUND_UP))
        min_point = float(Decimal(_min).quantize(Decimal('.1'),rounding=ROUND_DOWN))

        if self.debug:
            print('{} - {} / {} - {}'.format(_min,_max))
            print('{} - {} / {} - {}'.format(min_point,max_point))

        return {'max':max_point, 'min':min_point,}

    def get_range(self):
        if not self.target_lines:
            raise MyException('Aggregation data not found.')

        _dict = self._get_range_base(self.target_lines)
        if self.debug:
            for line in self.target_lines:
                print(line)
            print(_dict)

        return _dict


# Make Graph
class MkGraph(object):
    def __init__(self, glaph_dict, logger=None, debug=False):
        self.logger = logger or getLogger(__name__)
        self.debug = debug
        # figのインスタンスを作成
        self.fig = plt.figure(figsize=(15,10))
        # ax1のインスタンスを作成
        self.ax1 = self.fig.add_subplot()
        # graphの表示フォーマットの設定
        self.setup(glaph_dict)

    def set_format_xrange(self, form):
        # 時刻のフォーマットを設定
        _f = form or '%H:%M'
        xfmt = mpl.dates.DateFormatter(_f)
        self.fig.gca().xaxis.set_major_formatter(xfmt)

    def set_format_yrange(self, form):
        _f = form or '%.3f' # Default: y軸小数点以下3桁表示
        self.fig.gca().yaxis.set_major_formatter(plt.FormatStrFormatter(_f))

    def set_xlocator(self, interval=15):
        # ロケータで刻み幅を設定
        xloc = mpl.dates.MinuteLocator(byminute=range(0,60,interval))
        self.fig.gca().xaxis.set_major_locator(xloc)

    def set_xlim(self, start, end): # datetime
        self.ax1.set_xlim(start, end)

    def set_ylim(self, start, end, interval=0.100): # float
        self.ax1.set_ylim(start,end)
        self.ax1.set_yticks(np.arange(start,(end+0.001),interval))

    def setup(self, glaph_dict):
        x_format = glaph_dict['x_format'] if 'x_format' in glaph_dict else ''
        self.set_format_xrange(x_format)

        y_format = glaph_dict['y_format'] if 'y_format' in glaph_dict else ''
        self.set_format_yrange(y_format)

        x_interval = glaph_dict['x_interval'] if 'x_interval' in glaph_dict else 15
        self.set_xlocator(x_interval)

        if 'xlim' in glaph_dict:
            gd = glaph_dict['xlim']
            self.set_xlim(gd['start'],gd['end'])

        if 'ylim' in glaph_dict:
            gd = glaph_dict['ylim']
            interval = glaph_dict['y_interval'] if 'y_interval' in glaph_dict else 0.1
            self.set_ylim(gd['start'],gd['end'],interval)

    def _get_color(self, name):
        if '#' in name:
            return name
        color = {'yello':'y','magenta':'m','black':'k','white':'w'}
        return color[name] if name in color else 'y'

    def _get_symbol(self, keyword):
        symbol = {'long':'^','short':'v','line':'_','dot':'.'}
        return symbol[keyword] if keyword in symbol else '.'

    def set_value(self, xvals, yvals, color='yello', mark='dot', size=15): # x:[datetime] y:[float]
        self.ax1.plot(xvals, yvals, self._get_symbol(mark), markersize=size, color=self._get_color(color))

    def show(self, file_name='result.png'):
        self.fig.savefig(file_name,transparent=True)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Version %s\nThis script is...' % version,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--start', dest='start_range', action='store', metavar='YYYY/MM/DD_HH:MM:SS',default='',
                                   help='Aggregation range start')
    parser.add_argument('--end', dest='end_range', action='store', metavar='YYYY/MM/DD_HH:MM:SS',default='',
                                 help='Aggregation range end')
    parser.add_argument('--day', dest='day', action='store', metavar='YYYY/MM/DD',default='',
                                 help='Aggregation range is day')
    parser.add_argument('--segment', dest='segment', action='store', metavar='NUM m or h',default='5m',
                                     help='Time segment to aggregate (Default: 5m)')
    parser.add_argument('--dsp-segment', dest='dsp_segment', action='store', metavar='NUM', type=int, default='30',
                                         help='Display time minute (Default: 30 minute)')
    parser.add_argument('--pips', dest='pips', action='store', metavar='NUM',default='10',
                                  help='Pips segment to aggregate (Default: 10 pips)')
    parser.add_argument('--csv', dest='csv_file', action='store', metavar='FILE_NAME',default='yakujo.csv',
                                  help='Csv file name to read (Default: yakujo.csv)')
    parser.add_argument('--out', dest='out_file', action='store', metavar='FILE_NAME',default='result.png',
                                  help='Output image file name (Default: result.png)')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='Debug')
    parser.add_argument('--version', action='version', version='Version: %s' % version)

    args = parser.parse_args()
    return args

def get_aggregation_range(args):
    _dict = {'start':'','end':''}
    if args.day:
        _dict['start'] = '{} 00:00:00'.format(args.day)
        _dict['end'] = '{} 23:59:59'.format(args.day)
        return _dict

    if args.start_range:
        args.start_range = args.start_range.replace('_',' ')
        _dict['start'] = args.start_range
        if not args.end_range:
            dt = datetime.datetime.strptime(args.start_range, '%Y/%m/%d %H:%M:%S')
            _dict['end'] = '{} 23:59:59'.format(dt.strftime('%Y/%m/%d'))
            return _dict

    if args.end_range:
        args.end_range = args.end_range.replace('_',' ')
        _dict['end'] = args.end_range
        if not args.start_range:
            dt = datetime.datetime.strptime(args.end_range, '%Y/%m/%d %H:%M:%S')
            _dict['start'] = '{} 00:00:00'.format(dt.strftime('%Y/%m/%d'))
            return _dict

    return _dict

def get_color(kind, division):
    if kind == 'short':
        return '#00ff00' if division == 'start' else '#ffff33'
    if kind == 'long':
        return '#00ff00' if division == 'start' else '#ffff33'
    return '#333333'

def get_start_color(kind):
    return get_color(kind, 'start')

def get_end_color(kind):
    return get_color(kind, 'end')

def main():
    args = parse_args()
    try:
        lion = LionFX(args.csv_file,logger,args.debug)
        agg_range = get_aggregation_range(args)
        if not agg_range['start'] and not agg_range['end']:
            raise MyException('Incorrct aggregation range')

        result_data = lion.get_aggregate_data(agg_range['start'], agg_range['end'], args.segment)
        point_range = lion.get_range()
        logger.info('Glaph range: {} - {} ({}) / {} - {}'.format(
                        agg_range['start'],agg_range['end'],args.dsp_segment,
                        point_range['min'],point_range['max']))
        glaph_dict = {'xlim':{'start':lion.get_timedate(agg_range['start']),
                            'end':lion.get_timedate(agg_range['end'])},
                    'ylim':{'start':point_range['min'],
                            'end':point_range['max']},
                    'x_interval':args.dsp_segment,
                    'y_interval':float(args.pips)/100
        }

        graph = MkGraph(glaph_dict,logger,args.debug)
        logger.info('Start drawing the graph')
        for r in result_data:
            graph.set_value(r['start_time'], r['start_point'], get_start_color(r['kind']), r['kind'])
            graph.set_value(r['end_time'], r['end_point'], get_end_color(r['kind']), r['kind'])
        graph.show(args.out_file)
        logger.info('Completed drawing the graph: {}'.format(args.out_file))
    except MyException as e:
        logger.error(e.message)

if __name__ == '__main__':
    main()
