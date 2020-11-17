import xml.etree.ElementTree as ET


def load_video_annotations(path):
    root = ET.parse(path).getroot()
    task = root.find('meta/task')
    size = int(task.findtext('size'))
    width = int(task.findtext('original_size/width'))
    height = int(task.findtext('original_size/height'))
    result = [[] for _ in range(size)]
    for track in root.iterfind('track'):
        hand = track.get('label')
        for point in track.iterfind('points'):
            frame = int(point.get('frame'))
            abs_x, abs_y = map(float, point.get('points').split(','))
            result[frame].append({
                'label': 'open_pinch' if point.findtext("attribute[@name='open']") == 'true' else 'closed_pinch',
                'hand': hand,
                'x': abs_x / width,
                'y': abs_y / height,
            })
    return result
