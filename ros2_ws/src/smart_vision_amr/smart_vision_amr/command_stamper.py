"""Stamp Nav2's local raw Twist at its PC forwarding boundary.

Raw Twist has no creation timestamp. These stamps describe when this bridge
receives/forwards the local command, allowing the robot to reject transport
delay and replay; they cannot detect delay before a raw Twist reaches here.
"""

import json
import math
import uuid


class CommandStamperCore:
    def __init__(self, source_id=None):
        self.source_id = source_id or 'nav:'+uuid.uuid4().hex
        self.sequence = 0

    def packet(self, linear, angular, now_ns):
        if (type(now_ns) is not int or now_ns < 0
                or isinstance(linear, bool) or isinstance(angular, bool)
                or not isinstance(linear, (int, float))
                or not isinstance(angular, (int, float))
                or not math.isfinite(linear) or not math.isfinite(angular)):
            raise ValueError('Finite velocity and a nonnegative integer timestamp required')
        self.sequence += 1
        return {'source_id': self.source_id, 'seq': self.sequence,
                'sent_at_ns': now_ns, 'linear': float(linear), 'angular': float(angular)}


def main(args=None):
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    class CommandStamper(Node):
        def __init__(self):
            super().__init__('command_stamper')
            self.core = CommandStamperCore()
            # Velocity packets are short-lived: an old reliable DDS packet can
            # arrive after the mux's age limit and stall motion. The mux checks
            # sequence, stamp, and a short dead-man timeout on every packet.
            fast_control = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.publisher = self.create_publisher(
                String, '/smart_vision_amr/navigation_control', fast_control)
            self.create_subscription(Twist, '/smart_vision_amr/navigation_cmd_vel',
                                     self.forward, 1)

        def forward(self, message):
            try:
                packet = self.core.packet(message.linear.x, message.angular.z,
                                          self.get_clock().now().nanoseconds)
            except ValueError as exc:
                self.get_logger().warning(str(exc))
                return
            self.publisher.publish(String(data=json.dumps(packet, separators=(',', ':'))))

    rclpy.init(args=args)
    node = CommandStamper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
