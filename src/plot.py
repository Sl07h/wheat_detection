from wds import WheatDetectionSystem

wds = WheatDetectionSystem('Field_1_2', '1_08_10')

wds.create_map()
wds.draw_protocol()
wds.draw_tiles(['src'])
wds.save_map()
