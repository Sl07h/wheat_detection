from wds import WheatDetectionSystem

wds = WheatDetectionSystem('seedlings_2019', '06_03', 'frcnn', '400')
wds.read_metadata()

wds.create_map()
wds.draw_protocol()
wds.draw_images_on_map(False)
wds.draw_vegetation([0.1, 0.25, 0.5])
wds.save_map()
