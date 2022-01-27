from wds import WheatDetectionSystem

wds = WheatDetectionSystem('test_set1', '07_25', 'frcnn', '400')
wds.read_metadata()

wds.create_map()
wds.draw_protocol()
wds.draw_images_on_map(True)
wds.save_map()
