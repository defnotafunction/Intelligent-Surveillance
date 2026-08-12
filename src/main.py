from .camera import Camera

camera = Camera(
    curfew_start_hour=9,
    curfew_duration_in_hours=1
    )

if __name__ == '__main__':
    camera.run()