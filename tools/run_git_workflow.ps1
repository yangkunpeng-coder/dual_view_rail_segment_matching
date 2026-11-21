# 钢轨项目本机 Git 工作流
# develop 用于日常开发，main 用于稳定版本。

$ErrorActionPreference = "Stop"

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) {
        throw "Git 命令失败（退出码 $LASTEXITCODE） git $($args -join ' ')"
    }
}

function Invoke-GitWithMessage {
    param([string[]]$GitArguments, [string]$Message)

    $messageFile = [System.IO.Path]::GetTempFileName()
    try {
        $encoding = [System.Text.UTF8Encoding]::new($false)
        [System.IO.File]::WriteAllText($messageFile, $Message, $encoding)
        Invoke-Git @GitArguments -F $messageFile
    }
    finally {
        Remove-Item -LiteralPath $messageFile
    }
}

function Test-GitRef {
    param([string]$Ref)

    & git rev-parse --verify --quiet $Ref > $null
    return $LASTEXITCODE -eq 0
}

function Get-CurrentBranch {
    Invoke-Git branch --show-current
}

function Test-WorkingTreeChanges {
    return [bool](Invoke-Git status --porcelain)
}

function Assert-CleanTree {
    if (Test-WorkingTreeChanges) {
        Invoke-Git status --short
        throw "请先提交或暂存当前修改。"
    }
}

function Switch-LocalBranch {
    param([string]$Branch)

    if ((Get-CurrentBranch) -ceq $Branch) {
        return
    }

    Assert-CleanTree
    if (-not (Test-GitRef "refs/heads/$Branch")) {
        if ($Branch -ceq "main") {
            throw "首次发布时会从 develop 建立 main。"
        }
        throw "缺少本地 $Branch 分支。"
    }
    Invoke-Git switch --no-guess --no-overwrite-ignore $Branch
}

function Get-VersionTags {
    Invoke-Git tag --list --sort=-version:refname |
        Where-Object { $_ -cmatch '^v[0-9]+\.[0-9]+\.[0-9]+$' }
}

function Get-NextVersionTag {
    $latest = Get-VersionTags | Select-Object -First 1
    if (-not $latest) {
        return "v1.0.0"
    }

    $major, $minor, $patch = $latest.Substring(1).Split(".")
    return "v$major.$minor.$([long]$patch + 1)"
}

function Show-Status {
    Write-Host "项目目录 $repoRoot"
    Invoke-Git status --short --branch
    Invoke-Git branch --verbose --list develop main

    $latest = Get-VersionTags | Select-Object -First 1
    if ($latest) {
        Write-Host "最新版本 $latest"
    }
    if (Test-GitRef "HEAD") {
        Invoke-Git --no-pager log --oneline --decorate -8
    }
}

function Save-Develop {
    if ((Get-CurrentBranch) -cne "develop") {
        throw "请先切换到 develop 再提交。"
    }
    if (-not (Test-WorkingTreeChanges)) {
        Write-Host "工作区已干净。"
        return
    }

    Invoke-Git status --short
    $message = Read-Host "输入本次提交说明"
    if ([string]::IsNullOrWhiteSpace($message)) {
        Write-Host "已取消提交。"
        return
    }

    Invoke-Git var GIT_AUTHOR_IDENT | Out-Null
    Invoke-Git var GIT_COMMITTER_IDENT | Out-Null
    Invoke-Git add -A -- .
    Invoke-GitWithMessage -GitArguments @("commit") -Message $message
}

function Read-ReleaseMessage {
    param([string]$Tag)

    $title = Read-Host "版本标题（可留空）"
    if ([string]::IsNullOrWhiteSpace($title)) {
        $title = "Release $Tag"
    }

    Write-Host "逐行输入版本说明，空行结束。"
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add($title)
    $lines.Add("")
    while ($true) {
        $line = Read-Host
        if ([string]::IsNullOrWhiteSpace($line)) {
            break
        }
        $lines.Add($line)
    }
    return ($lines -join [char]10).TrimEnd()
}

function Release-Version {
    Assert-CleanTree
    $developRef = "refs/heads/develop"
    $mainRef = "refs/heads/main"
    if (-not (Test-GitRef $developRef)) {
        throw "请先在 develop 完成首次提交。"
    }

    $suggested = Get-NextVersionTag
    $tag = ([string](Read-Host "输入版本号（直接回车使用 $suggested）")).Trim()
    if (-not $tag) {
        $tag = $suggested
    }
    if ($tag -cnotmatch '^v?[0-9]+\.[0-9]+\.[0-9]+$') {
        throw "请输入三段式版本号，例如 v1.0.0。"
    }
    if (-not $tag.StartsWith("v")) {
        $tag = "v$tag"
    }
    if (Test-GitRef "refs/tags/$tag") {
        throw "本地标签 $tag 已存在。"
    }

    $message = Read-ReleaseMessage $tag
    $confirmation = Read-Host "将 develop 发布到 main 并创建 $tag？(Y/N)"
    if ($confirmation -notmatch '^y$') {
        Write-Host "已取消发布。"
        return
    }

    Assert-CleanTree
    Invoke-Git var GIT_COMMITTER_IDENT | Out-Null
    $developCommit = Invoke-Git rev-parse --verify $developRef
    $hasMain = Test-GitRef $mainRef
    if ($hasMain) {
        & git merge-base --is-ancestor $mainRef $developCommit
        if ($LASTEXITCODE -ne 0) {
            throw "main 无法快进到 develop，请先处理本地分支历史。"
        }
        Switch-LocalBranch "main"
    }
    else {
        Invoke-Git switch --no-overwrite-ignore -c main $developCommit
    }

    try {
        if ($hasMain) {
            Invoke-Git merge --ff-only --no-overwrite-ignore $developCommit
        }
        if ((Invoke-Git rev-parse --verify HEAD) -ne $developCommit) {
            throw "main 与待发布提交不一致，已停止发布。"
        }
        Invoke-GitWithMessage -GitArguments @("tag", "-a", $tag) -Message $message
        Write-Host "已在本机发布 $tag。" -ForegroundColor Green
    }
    finally {
        Switch-LocalBranch "develop"
    }
}

function Read-VersionTag {
    $tags = @(Get-VersionTags)
    if ($tags.Count -eq 0) {
        Write-Host "当前没有版本标签。"
        return
    }

    Write-Host ($tags -join ", ")
    $tag = ([string](Read-Host "输入版本标签，留空返回")).Trim()
    if ($tag -and $tag -cnotin $tags) {
        throw "找不到本地版本标签 $tag。"
    }
    return $tag
}

function Switch-OldVersion {
    Assert-CleanTree
    $tag = Read-VersionTag
    if ($tag) {
        Invoke-Git switch --detach --no-overwrite-ignore "refs/tags/$tag"
    }
}

function Save-Stash {
    if (Test-WorkingTreeChanges) {
        Invoke-Git stash push --include-untracked
    }
    else {
        Write-Host "没有需要暂存的修改。"
    }
}

function Restore-Stash {
    Assert-CleanTree
    if (-not (Test-GitRef "refs/stash")) {
        Write-Host "当前没有暂存记录。"
        return
    }

    Invoke-Git stash list
    if ((Read-Host "恢复最近一次暂存？(Y/N)") -match '^y$') {
        Invoke-Git stash pop
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not (Get-Command git -CommandType Application -ErrorAction SilentlyContinue)) {
    Write-Host "找不到 Git，请安装 Git for Windows。" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot ".git"))) {
    Write-Host "请先在 $repoRoot 执行 git init -b develop。" -ForegroundColor Yellow
    exit 1
}

Push-Location -LiteralPath $repoRoot
try {
    $gitRoot = (Resolve-Path -LiteralPath (Invoke-Git rev-parse --show-toplevel)).Path
    if ($gitRoot -ne $repoRoot) {
        throw "Git 仓库根目录与脚本所在项目不一致。"
    }

    while ($true) {
        $branch = Get-CurrentBranch
        if (-not $branch) {
            $branch = "分离 HEAD"
        }

        Write-Host ""
        Write-Host "钢轨项目本机 Git 工作流 | 当前 $branch" -ForegroundColor Cyan
        Write-Host "1. 查看状态和最近提交"
        Write-Host "2. 切换到 develop"
        Write-Host "3. 提交 develop 的开发进度"
        Write-Host "4. 发布稳定版本到 main 并创建标签"
        Write-Host "5. 切换到 main"
        Write-Host "6. 临时切换到旧版本"
        Write-Host "7. 暂存未提交修改"
        Write-Host "8. 恢复最近一次暂存"
        Write-Host "9. 查看版本说明"
        Write-Host "0. 退出"

        $choice = Read-Host "请选择"
        if ($null -eq $choice -or $choice -eq "0") {
            break
        }
        try {
            switch ($choice) {
                "1" { Show-Status }
                "2" { Switch-LocalBranch "develop" }
                "3" { Save-Develop }
                "4" { Release-Version }
                "5" { Switch-LocalBranch "main" }
                "6" { Switch-OldVersion }
                "7" { Save-Stash }
                "8" { Restore-Stash }
                "9" {
                    $tag = Read-VersionTag
                    if ($tag) {
                        Invoke-Git --no-pager show --no-patch "refs/tags/$tag"
                    }
                }
                default { Write-Host "请输入有效选项。" -ForegroundColor Yellow }
            }
        }
        catch {
            Write-Host $_.Exception.Message -ForegroundColor Red
        }
        [void](Read-Host "按 Enter 返回菜单")
    }
}
finally {
    Pop-Location
}
