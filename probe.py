"""Bounded setup checks. No learner, pretrained policy, or driving inputs."""
import argparse
import json
import math
from pathlib import Path
import socket
import struct
import time

FIELDS = ('speed', 'distance', 'x', 'y', 'z', 'steer', 'gas', 'brake',
          'finished', 'gear', 'rpm')
PACKET = struct.Struct('<11f')


def read_packet(sock, deadline):
    data = bytearray()
    while len(data) < PACKET.size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('No complete telemetry packet before deadline')
        sock.settimeout(remaining)
        chunk = sock.recv(PACKET.size - len(data))
        if not chunk:
            raise ConnectionError('Openplanet closed the connection')
        data.extend(chunk)
    values = PACKET.unpack(data)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Non-finite telemetry')
    return dict(zip(FIELDS, values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=('gpu', 'gamepad', 'telemetry'))
    parser.add_argument('--seconds', type=float, default=5)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error('--seconds must be in (0, 60]')
    if args.check == 'gpu':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f'CUDA unavailable in torch {torch.__version__}')
        x = torch.randn(256, 256, device='cuda', requires_grad=True)
        (x @ x.T).square().mean().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(x.grad).all()
        result = dict(torch=torch.__version__, cuda=torch.version.cuda,
                      device=torch.cuda.get_device_name(0), backward='passed')
    elif args.check == 'gamepad':
        import vgamepad
        pad = vgamepad.VX360Gamepad()
        try:
            pad.reset()
            pad.update()
            result = dict(virtual_controller='connected', driving_inputs_sent=False)
        finally:
            pad.reset()
            pad.update()
    else:
        samples = []
        start = time.monotonic()
        with socket.create_connection(('127.0.0.1', 9000), timeout=3) as sock:
            deadline = start + args.seconds
            while time.monotonic() < deadline:
                try:
                    frame = read_packet(sock, deadline)
                except TimeoutError:
                    break
                frame['received_seconds'] = time.monotonic() - start
                samples.append(frame)
        if not samples:
            raise RuntimeError('Connected but no telemetry. Load the map and enter driving mode.')
        result = dict(frames=len(samples), first=samples[0], last=samples[-1],
                      max_speed=max(s['speed'] for s in samples))
        result['samples'] = samples
    runs = Path(__file__).resolve().parent / 'runs'
    runs.mkdir(exist_ok=True)
    path = runs / f'{args.check}-{time.time_ns()}.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
