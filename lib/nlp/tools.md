```
tail -f error.log
less +F error.log
journalctl -u <service> -f

/etc/nginx/sites-available/<site[same as service name]>

/etc/systemd/system/<service>.service
sudo systemct daemon-reload
sudo systemctl status|start|stop|restart <service>
```