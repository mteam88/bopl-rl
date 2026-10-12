/* Virtual Xbox 360 pad on /dev/uinput for driving the real game headless
 * (eval/shadow_live/session.sh). The game's character select only lets a
 * keyboard join while an unjoined gamepad is present, and a 3-player round
 * needs two human devices, so the live shadow test brings its own pads.
 *
 *   vpad [name]      reads commands from stdin, one per line:
 *     <button> <0|1>   a b x y lb rb back start guide ls rs
 *     lx|ly|rx|ry <v>  stick axis, -32768..32767
 *     lt|rt <v>        trigger, 0..255
 *     hx|hy <v>        dpad, -1..1
 *   Each line is one event plus a sync. EOF removes the device.
 *
 *   cc -O2 -o vpad eval/shadow_live/vpad.c */
#include <fcntl.h>
#include <linux/uinput.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static int fd;

static void emit(int type, int code, int value) {
    struct input_event e;
    memset(&e, 0, sizeof e);
    e.type = (unsigned short)type; e.code = (unsigned short)code; e.value = value;
    if (write(fd, &e, sizeof e) != sizeof e) perror("vpad write");
}

static void axis(int code, int min, int max, int fuzz, int flat) {
    struct uinput_abs_setup a;
    memset(&a, 0, sizeof a);
    a.code = (unsigned short)code;
    a.absinfo.minimum = min; a.absinfo.maximum = max; a.absinfo.fuzz = fuzz; a.absinfo.flat = flat;
    if (ioctl(fd, UI_SET_ABSBIT, code) || ioctl(fd, UI_ABS_SETUP, &a)) perror("vpad axis");
}

int main(int argc, char **argv) {
    fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) { perror("/dev/uinput"); return 1; }
    static const struct { const char *name; int code; } buttons[] = {
        {"a", BTN_SOUTH}, {"b", BTN_EAST}, {"x", BTN_NORTH}, {"y", BTN_WEST}, {"lb", BTN_TL}, {"rb", BTN_TR},
        {"back", BTN_SELECT}, {"start", BTN_START}, {"guide", BTN_MODE}, {"ls", BTN_THUMBL}, {"rs", BTN_THUMBR}};
    static const struct { const char *name; int code; } axes[] = {
        {"lx", ABS_X}, {"ly", ABS_Y}, {"rx", ABS_RX}, {"ry", ABS_RY}, {"lt", ABS_Z}, {"rt", ABS_RZ},
        {"hx", ABS_HAT0X}, {"hy", ABS_HAT0Y}};
    ioctl(fd, UI_SET_EVBIT, EV_KEY);
    ioctl(fd, UI_SET_EVBIT, EV_ABS);
    for (size_t i = 0; i < sizeof buttons / sizeof *buttons; i++) ioctl(fd, UI_SET_KEYBIT, buttons[i].code);
    axis(ABS_X, -32768, 32767, 16, 128); axis(ABS_Y, -32768, 32767, 16, 128);
    axis(ABS_RX, -32768, 32767, 16, 128); axis(ABS_RY, -32768, 32767, 16, 128);
    axis(ABS_Z, 0, 255, 0, 0); axis(ABS_RZ, 0, 255, 0, 0);
    axis(ABS_HAT0X, -1, 1, 0, 0); axis(ABS_HAT0Y, -1, 1, 0, 0);
    struct uinput_setup setup;
    memset(&setup, 0, sizeof setup);
    setup.id.bustype = BUS_USB; setup.id.vendor = 0x045e; setup.id.product = 0x028e; setup.id.version = 0x110;
    snprintf(setup.name, UINPUT_MAX_NAME_SIZE, "%s", argc > 1 ? argv[1] : "Microsoft X-Box 360 pad");
    if (ioctl(fd, UI_DEV_SETUP, &setup) || ioctl(fd, UI_DEV_CREATE)) { perror("vpad create"); return 1; }
    printf("ready\n"); fflush(stdout);
    char line[128], name[32];
    int value;
    while (fgets(line, sizeof line, stdin)) {
        if (sscanf(line, "%31s %d", name, &value) != 2) continue;
        int done = 0;
        for (size_t i = 0; i < sizeof buttons / sizeof *buttons && !done; i++)
            if (!strcmp(name, buttons[i].name)) { emit(EV_KEY, buttons[i].code, value != 0); done = 1; }
        for (size_t i = 0; i < sizeof axes / sizeof *axes && !done; i++)
            if (!strcmp(name, axes[i].name)) { emit(EV_ABS, axes[i].code, value); done = 1; }
        if (done) emit(EV_SYN, SYN_REPORT, 0);
    }
    ioctl(fd, UI_DEV_DESTROY);
    close(fd);
    return 0;
}
