from wds import WheatDetectionSystem

wds = WheatDetectionSystem('seedlings_2019', '06_03_test', 'frcnn', '400', 0.2)
wds.read_metadata()
wds.create_map()
wds.draw_protocol()
wds.draw_tiles()
wds.save_map()