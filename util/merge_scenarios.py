# merge_rou.py
import xml.etree.ElementTree as ET

def merge_rou(normal_path, peak_path, out_path, id_offset=8000, time_offset=3600.0):
    t1 = ET.parse(normal_path); r1 = t1.getroot()      # normal <routes>
    t2 = ET.parse(peak_path);   r2 = t2.getroot()      # peak   <routes>

    for veh in r2.findall('vehicle'):
        # shift id
        if 'id' in veh.attrib:
            veh.attrib['id'] = str(int(veh.attrib['id']) + id_offset)
        # shift depart
        if 'depart' in veh.attrib:
            veh.attrib['depart'] = str(float(veh.attrib['depart']) + time_offset)

        # append to normal
        r1.append(veh)

    t1.write(out_path, encoding='utf-8', xml_declaration=True)