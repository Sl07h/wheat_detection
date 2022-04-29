from wds import WheatDetectionSystem

wds = WheatDetectionSystem('penza_2021', '04_26_3_meters', 'frcnn', '400')
wds.create_mask('tgi')

wds.read_metadata()
wds.create_map()
wds.draw_protocol()
wds.draw_vegetation()
wds.draw_tiles('src')
wds.draw_tiles('tgi')
wds.save_map()
