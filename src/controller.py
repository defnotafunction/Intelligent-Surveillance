import logging
import pygame

pygame.init()
pygame.joystick.init()

def get_any_controllers_connected() -> bool:
    return pygame.joystick.get_count() > 0

class ControllerManager:
    def __init__(self, controller_id: int = 0) -> None:
        try:
            self.controller_id = controller_id
            self.controller = pygame.joystick.Joystick(self.controller_id)
        except:
            raise ControllerManager(controller_id=controller_id)

    def listen(self) -> None:
        """Listens for buttons clicks."""
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    return 'QUIT'

                elif event.button == 1:
                    return 'REMEMBER'

                elif event.button == 2:
                    return 'COUNT'

    def kill(self) -> None:
        """Shuts down all pygame modules."""
        pygame.joystick.quit()
        pygame.quit()
