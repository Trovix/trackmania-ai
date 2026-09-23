"""Short, supervised local-map control test. Always releases inputs on exit."""
import argparse
import json
import math
from pathlib import Path
import socket
import time
import win32gui
import vgamepad
from probe import read_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brake', action='store_true')
    args = parser.parse_args()
    hwnd = win32gui.FindWindow(None, 'Trackmania')
    def check_focus():
        if not hwnd or win32gui.IsIconic(hwnd) or win32gui.GetForegroundWindow() != hwnd:
            raise RuntimeError('Trackmania must remain foreground; inputs released')
    check_focus()
    samples = []
    pad = vgamepad.VX360Gamepad()
    start = time.monotonic()
    result = {}
    try:
        with socket.create_connection(('127.0.0.1', 9000), timeout=3) as sock:
            def observe(phase, seconds):
                deadline = time.monotonic() + seconds
                frames = []
                while time.monotonic() < deadline:
                    check_focus()
                    try:
                        frame = read_packet(sock, deadline)
                    except TimeoutError:
                        break
                    frame.update(phase=phase, received_seconds=time.monotonic()-start)
                    frames.append(frame)
                    samples.append(frame)
                if not frames:
                    raise RuntimeError('No telemetry; inputs released')
                return frames
            pad.reset()
            pad.update()
            baseline = observe('baseline', 0.5)[-1]
            if abs(baseline['speed']) > 1 or baseline['finished']:
                raise RuntimeError('Expected stationary car in an active race')
            pad.right_trigger_float(1.0)
            pad.update()
            observe('throttle', 1.2)
            pad.left_joystick_float(0.25, 0.0)
            pad.update()
            observe('steer', 0.35)
            pad.reset()
            pad.update()
            released = observe('release', 0.4)[-1]
            if args.brake:
                pad.left_trigger_float(1.0)
                pad.update()
                braking = observe('brake', 0.8)
                pad.reset()
                pad.update()
                brake_released = observe('brake_release', 0.25)[-1]
            pad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
            pad.update()
            observe('reset_press', 0.1)
            pad.reset()
            pad.update()
            final = observe('after_reset', 4.0)[-1]
            delta = math.dist([baseline[k] for k in ('x','y','z')],
                              [final[k] for k in ('x','y','z')])
            movement = max(math.dist([baseline[k] for k in ('x','y','z')],
                                    [s[k] for k in ('x','y','z')]) for s in samples)
            checks = dict(throttle=any(s['gas'] > 0.1 and s['speed'] > 1 for s in samples),
                          steering=any(abs(s['steer']) > 0.1 for s in samples if s['phase']=='steer'),
                          release=abs(released['gas']) < 0.01 and abs(released['steer']) < 0.01,
                          reset=movement > 1 and delta < 0.5 and abs(final['speed']) < 1)
            result.update(checks=checks, max_speed=max(s['speed'] for s in samples),
                          max_displacement=movement, reset_position_error=delta, baseline=baseline, final=final)
            if args.brake:
                checks['brake_input'] = any(s['brake'] > 0.5 for s in braking)
                checks['brake_release'] = brake_released['brake'] == 0
                result['braking_speed_before'] = released['speed']
                result['braking_speed_after'] = braking[-1]['speed']
                checks['slowed_during_braking'] = braking[-1]['speed'] < released['speed'] - 1
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        pad.reset()
        pad.update()
        result['samples'] = samples
        out = Path(__file__).resolve().parent / 'runs' / f'controls-{time.time_ns()}.json'
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}, indent=2))
        print(out)


if __name__ == '__main__':
    main()
